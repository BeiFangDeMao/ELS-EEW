"""One active-output epoch adapter; original optimization, selection and RNG helpers."""
import logging
from contextlib import contextmanager
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from training_adapter import (atomic_json,atomic_torch,cpu_state,optimizer_scheduler,
                              rng_state,restore_rng,update_selection)
from training import Trainstudents as original
from ablation_models import loss,clamp_active,task_maes


def checked_clip(model, maximum):
    """Preserve ordinary clipping; distinguish bad elements from FP32 norm overflow."""
    parameters=[p for p in model.parameters() if p.grad is not None]
    bad=[n for n,p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    if bad: raise FloatingPointError('Nonfinite gradient elements: '+', '.join(bad))
    # error_if_nonfinite raises BEFORE modifying any gradient.
    try:
        return float(torch.nn.utils.clip_grad_norm_(parameters,maximum,error_if_nonfinite=True))
    except RuntimeError as exc:
        norms=torch.stack([torch.linalg.vector_norm(p.grad.detach().double()) for p in parameters])
        total=torch.linalg.vector_norm(norms)
        if not torch.isfinite(total): raise FloatingPointError('Nonfinite FP64 gradient norm') from exc
        coefficient=(maximum/(total+1e-6)).clamp(max=1.)
        for p in parameters: p.grad.mul_(coefficient.to(p.grad.dtype))
        logging.warning('Finite gradient elements; FP32 norm overflow. FP64 norm=%g',float(total))
        return float(total)


@contextmanager
def precise_retry(device):
    """One numerical retry, keeping cuDNN enabled and all other flags unchanged."""
    old=torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32=False
        with torch.backends.cudnn.flags(enabled=torch.backends.cudnn.enabled,
                benchmark=torch.backends.cudnn.benchmark,
                deterministic=torch.backends.cudnn.deterministic,allow_tf32=False):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32=old


def epoch(model,loader,spec,cfg,device,optimizer=None,scheduler=None,diagnostic_dir=None,epoch_number=None):
    training=optimizer is not None
    model.train(training)
    losses=[];predictions=[];labels=[]
    with torch.set_grad_enabled(training):
        for batch_index,(wave,y,keys,_) in enumerate(tqdm(loader,desc=f'{spec["name"]} '+('train' if training else 'validation'))):
            wave,y=wave.to(device),y.to(device)
            original.require_finite(wave,f'input {keys[:2]}')
            original.require_finite(y,'labels')
            if training: optimizer.zero_grad()
            before=rng_state() if training else None
            raw,_=model(wave)
            original.require_finite(raw,'raw predictions')
            pred=clamp_active(raw,spec['task'])
            value=loss(model,pred,y,spec,cfg['uniform_time_weight'])
            original.require_finite(value,'loss')
            if training:
                value.backward()
                try:
                    checked_clip(model,cfg['grad_clip'])
                except FloatingPointError as exc:
                    # No optimizer/scheduler step has occurred. Preserve the failure,
                    # replay the SAME data once; never skip samples or zero bad grads.
                    stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                    snapshot=None
                    if diagnostic_dir is not None:
                        diagnostic_dir.mkdir(parents=True,exist_ok=True)
                        snapshot=diagnostic_dir/f'epoch{epoch_number}_batch{batch_index}_{stamp}.pth'
                        atomic_torch(snapshot,dict(epoch=epoch_number,batch_index=batch_index,keys=list(keys),
                            wave=wave.detach().cpu(),labels=y.detach().cpu(),model_state_dict=cpu_state(model),
                            optimizer_state_dict=optimizer.state_dict(),scheduler_state_dict=scheduler.state_dict(),
                            rng_before=before,loss=float(value.detach()),error=str(exc),
                            gradients={n:p.grad.detach().cpu() for n,p in model.named_parameters() if p.grad is not None},
                            cudnn_tf32=torch.backends.cudnn.allow_tf32,matmul_tf32=torch.backends.cuda.matmul.allow_tf32))
                    logging.warning('Epoch %s batch %d: %s. Retrying same batch once with TF32 disabled; snapshot=%s',
                                    epoch_number,batch_index,exc,snapshot)
                    restore_rng(before)
                    optimizer.zero_grad(set_to_none=True)
                    try:
                        with precise_retry(device):
                            raw,_=model(wave)
                            original.require_finite(raw,'retry raw predictions')
                            pred=clamp_active(raw,spec['task'])
                            value=loss(model,pred,y,spec,cfg['uniform_time_weight'])
                            original.require_finite(value,'retry loss')
                            value.backward()
                            checked_clip(model,cfg['grad_clip'])
                    except (FloatingPointError,RuntimeError) as retry_error:
                        if snapshot is not None:
                            atomic_json(snapshot.with_suffix('.json'),dict(status='retry_failed',error=str(retry_error)))
                        raise FloatingPointError(f'Same-batch precise retry failed; checkpoint unchanged; snapshot={snapshot}') from retry_error
                    if snapshot is not None:
                        atomic_json(snapshot.with_suffix('.json'),dict(status='retry_succeeded',
                            note='Same data/loss/weights; TF32 disabled for retry only. Backend dropout need not be bitwise identical.'))
                    logging.warning('Precise retry succeeded; applying exactly one optimizer/scheduler step.')
                optimizer.step()
                for param in model.parameters(): original.require_finite(param,'parameter after update')
                scheduler.step()  # Per successful batch, exactly original CyclicLR timing.
            losses.append(float(value.detach()))
            predictions.append(pred.detach().cpu().numpy().reshape(-1,pred.shape[-1]))
            labels.append(y.detach().cpu().numpy().repeat(pred.shape[1],axis=0))
    if not losses: raise ValueError('Empty training/validation loader')
    maes=task_maes(np.concatenate(predictions),np.concatenate(labels),spec['task'])
    if not np.isfinite(maes).all(): raise FloatingPointError('Nonfinite active-task MAE')
    return float(np.mean(losses)),maes


def fit(model,spec,cfg,loaders,folder,fingerprint,device):
    """Atomic epoch commits; complete RNG/AdamW/CyclicLR/selection resume."""
    model.to(device)
    optimizer,scheduler=optimizer_scheduler(model,cfg,len(loaders['train']))
    start=1;history=[];best=np.full(3 if spec['task']=='all' else 1,np.inf);bad=0;best_epoch=0;payload=None
    path=folder/'resume_checkpoint.pth'
    if path.exists():
        state=torch.load(path,map_location='cpu',weights_only=False)
        if state['fingerprint']!=fingerprint or state['spec']!=spec: raise RuntimeError('Resume identity changed')
        model.load_state_dict(state['model_state_dict'],strict=True)
        optimizer.load_state_dict(state['optimizer_state_dict']);scheduler.load_state_dict(state['scheduler_state_dict'])
        start=state['epoch']+1;history=state['history'];best=np.asarray(state['historical_best']);bad=state['bad_epochs']
        payload=state['best_payload'];best_epoch=state['best_epoch']
        restore_rng(state['rng'])
        logging.info('%s resume from committed epoch %d, bad=%d',spec['name'],state['epoch'],bad)
    for number in range(start,cfg['epochs']+1):
        if bad>=cfg['patience']: break
        logging.info('%s epoch %d/%d, bad=%d/%d',spec['name'],number,cfg['epochs'],bad,cfg['patience'])
        train_loss,train_maes=epoch(model,loaders['train'],spec,cfg,device,optimizer,scheduler,
                                   diagnostic_dir=folder/'gradient_diagnostics',epoch_number=number)
        val_loss,val_maes=epoch(model,loaders['validation'],spec,cfg,device)
        current=np.asarray(val_maes)
        if spec['task']=='all': best,bad,accepted=update_selection(best,current,bad,cfg['early_stop_epsilon'])
        else:
            accepted=bool(current[0]<best[0]-cfg['early_stop_epsilon'])
            best=current.copy() if accepted else best
            bad=0 if accepted else bad+1
        history.append(dict(epoch=number,train_loss=train_loss,val_loss=val_loss,train_maes=train_maes,val_maes=val_maes,
                            checkpoint_updated=accepted,bad_epochs=bad,learning_rates=[g['lr'] for g in optimizer.param_groups]))
        if accepted:
            best_epoch=number
            payload=dict(epoch=number,model_state_dict=cpu_state(model),validation_maes=val_maes,spec=spec,fingerprint=fingerprint)
        state=dict(epoch=number,model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),
                   scheduler_state_dict=scheduler.state_dict(),history=history,historical_best=best.tolist(),bad_epochs=bad,
                   best_payload=payload,best_epoch=best_epoch,rng=rng_state(),fingerprint=fingerprint,spec=spec)
        atomic_torch(path,state)
        if accepted: atomic_torch(folder/'best.pth',payload)
        atomic_json(folder/'training_history.json',history)
        pd.DataFrame(history).to_csv(folder/'training_history.csv',index=False)
        logging.info('%s epoch %d: train loss %.6f, val loss %.6f, val MAE=%s, saved=%s',
                     spec['name'],number,train_loss,val_loss,val_maes,accepted)
    if payload is None: raise RuntimeError('No best checkpoint available')
    # Also repair ancillary files if a crash followed the terminal commit.
    if not (folder/'best.pth').exists(): atomic_torch(folder/'best.pth',payload)
    model.load_state_dict(payload['model_state_dict'],strict=True)
    atomic_json(folder/'training_history.json',history)
    pd.DataFrame(history).to_csv(folder/'training_history.csv',index=False)
    return dict(best_epoch=best_epoch,completed_epochs=len(history),validation_maes=payload['validation_maes'],
                selection='at least 2 tasks improve; update improved historical entries only' if spec['task']=='all' else 'own-task MAE improves',
                early_stop_patience=cfg['patience'],reused=False)
