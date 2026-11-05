"""Original epoch/loss/evaluation functions plus epoch-boundary resume management."""
import copy
import json
import logging
import os
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from training import Trainstudents as original


def atomic_torch(path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    torch.save(obj, temp)
    os.replace(temp, path)


def atomic_json(path, obj):
    def clean(value):
        if isinstance(value, dict): return {str(k):clean(v) for k,v in value.items()}
        if isinstance(value, (tuple,list)): return [clean(v) for v in value]
        if isinstance(value, np.generic): return clean(value.item())
        if isinstance(value, Path): return str(value)
        if isinstance(value, float) and not np.isfinite(value): return None
        return value
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(clean(obj), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda']])


def configure_original(cfg, device):
    original.device = device
    original.FIXED_LENGTH = cfg['sequence_length']
    original.NUM_CHUNKS = cfg['sequence_length']//cfg['chunk_size']
    original.BATCH_SIZE = cfg['batch_size']
    # Refuse silently drifting imported training constants.
    for key, attr in [('weight_decay','WEIGHT_DECAY'), ('backbone_base_lr','BACKBONE_BASE_LR'),
                      ('backbone_max_lr','BACKBONE_MAX_LR'), ('head_base_lr','HEAD_BASE_LR'),
                      ('head_max_lr','HEAD_MAX_LR'), ('early_stop_epsilon','EARLY_STOP_EPS')]:
        if cfg[key] != getattr(original, attr):
            raise ValueError(f'{key} differs from original training code; no per-size tuning permitted')


def optimizer_scheduler(model, cfg, steps):
    backbone = list(model.feature_extractor.parameters())
    head = list(model.output_head.parameters())+[p for n,p in model.named_parameters() if 'log_sigma' in n]
    optimizer = torch.optim.AdamW([dict(params=backbone,lr=cfg['backbone_base_lr']),
                                  dict(params=head,lr=cfg['head_base_lr'])],
                                 weight_decay=cfg['weight_decay'], betas=(.9,.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
        base_lr=[cfg['backbone_base_lr'],cfg['head_base_lr']],
        max_lr=[cfg['backbone_max_lr'],cfg['head_max_lr']],
        step_size_up=cfg['cycle_up_epochs']*steps, step_size_down=cfg['cycle_down_epochs']*steps,
        mode='triangular2', cycle_momentum=False)
    return optimizer,scheduler


def validation_vector(metrics):
    return np.asarray([metrics['mag']['mae'],metrics['dis']['mae'],metrics['azi_deg']['mae']],dtype=float)


def update_selection(best, current, bad_epochs, epsilon):
    """Exactly original semantics: update improved entries only IF >=2 improve."""
    improved = current < best-epsilon
    accepted = int(improved.sum()) >= 2
    updated = np.where(improved, current, best) if accepted else best.copy()
    return updated, 0 if accepted else bad_epochs+1, accepted


class FailOnSkippedBatch(logging.Handler):
    """Original epoch code skips NaNs; this study must fail rather than mask them."""
    def emit(self, record):
        if '跳过异常批次' in record.getMessage():
            raise FloatingPointError('Scaling run aborted on invalid training batch: '+record.getMessage())


def cpu_state(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def train_model(model, spec, cfg, loaders, folder, fingerprint):
    """Resume at a committed epoch; interrupted partial epoch is replayed."""
    train_loader, val_loader = loaders
    latest = folder/'resume_checkpoint.pth'
    model.to(original.device)
    optimizer,scheduler = optimizer_scheduler(model,cfg,len(train_loader))
    history, best, bad, best_epoch, start, best_payload = [],np.full(3,np.inf),0,0,1,None
    if latest.exists():
        state = torch.load(latest,map_location='cpu',weights_only=False)
        if state['fingerprint'] != fingerprint:
            raise RuntimeError('Resume fingerprint mismatch; choose a new results directory')
        model.load_state_dict(state['model_state_dict'],strict=True)
        optimizer.load_state_dict(state['optimizer_state_dict'])
        scheduler.load_state_dict(state['scheduler_state_dict'])
        history,best,bad,best_epoch = state['history'],np.asarray(state['historical_bests']),state['bad_epochs'],state['best_epoch']
        best_payload = state['best_payload']
        start = state['epoch']+1
        restore_rng(state['rng'])
        logging.info('%s: full resume from committed epoch %d',spec['model'],state['epoch'])
        # Authoritative latest checkpoint includes best weights, avoiding a
        # two-file transaction race if a process died while updating best.pth.
        if best_payload is not None: atomic_torch(folder/'best.pth',best_payload)
    handler = FailOnSkippedBatch()
    logging.getLogger().addHandler(handler)
    try:
        for epoch in range(start,cfg['epochs']+1):
            if bad >= cfg['patience']: break
            logging.info('%s epoch %d/%d, early-stop counter %d/%d', spec['model'],epoch,cfg['epochs'],bad,cfg['patience'])
            loss,metrics,parts,_ = original.train_one_epoch(model,train_loader,optimizer,scheduler,epoch,
                grad_clip=cfg['grad_clip'],diagnostic_dir=str(folder/'diagnostics'),model_id=spec['model'])
            val_loss,val_metrics,val_parts,_ = original.evaluate(model,val_loader)
            current = validation_vector(val_metrics)
            if not np.isfinite(current).all(): raise FloatingPointError('Nonfinite validation metrics')
            best,bad,accepted = update_selection(best,current,bad,cfg['early_stop_epsilon'])
            row = dict(epoch=epoch,train_loss=loss,val_loss=val_loss,
                train_mag_mae=metrics['mag']['mae'],train_dis_mae=metrics['dis']['mae'],train_azi_maae=metrics['azi_deg']['mae'],
                val_mag_mae=current[0],val_dis_mae=current[1],val_azi_maae=current[2],
                train_task_losses=list(parts),val_task_losses=list(val_parts),checkpoint_updated=accepted,
                backbone_lr=optimizer.param_groups[0]['lr'],head_lr=optimizer.param_groups[1]['lr'],early_stop_counter=bad)
            history.append(row)
            if accepted:
                best_epoch=epoch
                best_payload = dict(epoch=epoch,model_id=spec['model'],architecture_config=model.architecture_config,
                    model_state_dict=cpu_state(model),validation_metrics=val_metrics, fingerprint=fingerprint)
            state=dict(epoch=epoch,model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),
                       scheduler_state_dict=scheduler.state_dict(),history=history,historical_bests=best.tolist(),
                       bad_epochs=bad,best_epoch=best_epoch,best_payload=best_payload,rng=rng_state(),fingerprint=fingerprint)
            atomic_torch(latest,state)
            if accepted: atomic_torch(folder/'best.pth',best_payload)
            pd.DataFrame(history).to_csv(folder/'training_history.csv',index=False)
            atomic_json(folder/'training_history.json',history)
            logging.info('%s epoch %d: loss %.6f val %.6f; MAE M %.6f D %.6f Az %.6f; saved=%s',
                         spec['model'],epoch,loss,val_loss,*current,accepted)
    finally:
        logging.getLogger().removeHandler(handler)
    if best_payload is None: raise RuntimeError('No valid best checkpoint exists')
    # Rebuild ancillary files even if interruption occurred after the terminal
    # checkpoint commit but before history/best files were written.
    atomic_torch(folder/'best.pth',best_payload)
    pd.DataFrame(history).to_csv(folder/'training_history.csv',index=False)
    atomic_json(folder/'training_history.json',history)
    model.load_state_dict(best_payload['model_state_dict'],strict=True)
    return dict(status='training_complete',completed_epochs=len(history),best_epoch=best_epoch,
                historical_bests=best.tolist(),best_validation=best_payload['validation_metrics'])


def inference_metrics(model, loader, cfg, split):
    """One causal full-window pass; checked against stream_step in preflight.

    Only five requested times retained (no multi-GB repeated-label arrays).
    Same finite predictions and original clipping; circular angle differences.
    """
    requested=cfg['report_times']
    indices=[int(np.floor(t*cfg['sampling_rate']/cfg['chunk_size']+1e-9))-1 for t in requested]
    if min(indices)<0 or max(indices)>=93: raise ValueError('Invalid report times')
    predictions,labels=[],[]
    model.eval()
    from tqdm import tqdm
    with torch.inference_mode():
        for wave,y,_,_ in tqdm(loader,desc=f'{split} fixed-time evaluation'):
            raw,_=model(wave.to(original.device))
            if raw.shape != (len(wave),93,4) or not torch.isfinite(raw).all():
                raise FloatingPointError('Bad inference output; refusing to replace NaN/Inf with fake predictions')
            out=original.safe_clamp_preds(raw)
            predictions.append(out[:,indices,:].cpu().numpy())
            labels.append(y.numpy())
    p=np.concatenate(predictions).astype(np.float64)
    y=np.concatenate(labels).astype(np.float64)
    output=[]
    for j,(target,index) in enumerate(zip(requested,indices)):
        dm=p[:,j,0]-y[:,0]; dd=p[:,j,1]-y[:,1]
        pa=np.degrees(np.arctan2(p[:,j,2],p[:,j,3]))%360
        ta=np.degrees(np.arctan2(y[:,2],y[:,3]))%360
        da=np.abs((pa-ta+180)%360-180)
        output.append(dict(split=split,requested_time_s=target,actual_output_time_s=(index+1)*.32,
                           output_chunk=index+1,records=len(y),magnitude_mae=float(np.abs(dm).mean()),
                           magnitude_rmse=float(np.sqrt((dm*dm).mean())),distance_mae_km=float(np.abs(dd).mean()),
                           distance_rmse_km=float(np.sqrt((dd*dd).mean())),azimuth_circular_mae_deg=float(da.mean()),
                           azimuth_median_ae_deg=float(np.median(da))))
    return output
