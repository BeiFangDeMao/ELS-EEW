"""IDE/Python entry point: BAZ-Network only by default, same saved cohort."""
import argparse
import json
import logging
import multiprocessing as mp
from pathlib import Path
import sys
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from models import create_model,task_for,loss_for,prediction_for
import train_comparison as shared
from comparison_dataset import ComparisonDataset,resolve_dataset,load_memory_bank


def components(output):
    return output if isinstance(output,(tuple,list)) else (output,)


def epoch_pass(model,data,name,device,optimizer=None,clip=.8,writer=None):
    model.train(optimizer is not None)
    total_loss=total_error=0.;count=0
    with torch.set_grad_enabled(optimizer is not None):
        for wave,target,keys in tqdm(data,desc='train' if optimizer else 'evaluate',leave=False):
            wave,target=wave.to(device),target.to(device)
            output=model(wave)
            tensors=components(output)
            expected=[(len(wave),2)] if name=='distpt_distance_only' else [(len(wave),1)]*2
            if [tuple(t.shape) for t in tensors]!=expected:raise ValueError('Invalid output shapes')
            loss=loss_for(name,output,target)
            if not all(torch.isfinite(t).all() for t in tensors) or not torch.isfinite(loss):
                raise FloatingPointError(f'{name}: nonfinite prediction/loss; no batch skipped')
            if optimizer:
                optimizer.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),clip,error_if_nonfinite=True)
                optimizer.step()
            prediction=prediction_for(name,output).detach()
            error=abs(prediction-target)
            if task_for(name)=='baz':error=torch.minimum(error%360,360-error%360)
            count+=len(wave);total_loss+=float(loss.detach())*len(wave);total_error+=float(error.sum())
            if writer:
                for key,y,p in zip(keys,target[:,0].tolist(),prediction[:,0].tolist()):writer.writerow([key,y,p])
    if not count:raise RuntimeError('Empty split')
    return dict(loss=total_loss/count,mae=total_error/count,records=count)


def preflight(cfg,directory=None):
    for name in cfg['models']:
        shared.seed_all(cfg['seed']);model=create_model(name).train()
        x=torch.randn(2,3,288);target=torch.full((2,1),50. if task_for(name)=='dis' else 180.)
        output=model(x);loss=loss_for(name,output,target)
        assert torch.isfinite(loss) and all(torch.isfinite(t).all() for t in components(output))
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        if name=='distpt_distance_only':
            assert all(p.grad is None and not p.requires_grad for p in model.backbone.lin_ptrvl.parameters())
        model.eval()
        with torch.inference_mode():
            zero=model(torch.zeros(2,3,288))
            assert all(torch.isfinite(t).all() for t in components(zero))
            if name=='baz_network':
                separate=model(x[:1]);mixed=model(torch.cat([x[:1],x[1:]*100],dim=0))
                for a,b in zip(separate,mixed):torch.testing.assert_close(a,b[:1],atol=2e-5,rtol=2e-5)
            if directory:
                dataset=ComparisonDataset(directory,'train',task_for(name))
                wave,_,key=dataset[0];dataset.close()
                output=model(torch.from_numpy(wave[None]))
                assert all(torch.isfinite(t).all() for t in components(output))
                print('Real training waveform check:',key,flush=True)
                if name=='baz_network':
                    checked=0
                    for split in ['train','validation','test']:
                        sample=ComparisonDataset(directory,split,'baz')
                        indexes=np.linspace(0,len(sample)-1,32,dtype=int)
                        batch=torch.from_numpy(np.stack([sample[int(i)][0] for i in indexes]))
                        sample.close()
                        cov=model._cov(batch)
                        features=model._compute_cov_and_eig(batch)
                        outputs=model(batch)
                        assert torch.isfinite(cov).all() and torch.isfinite(features).all()
                        assert all(torch.isfinite(t).all() for t in outputs)
                        assert torch.isfinite(prediction_for(name,outputs)).all()
                        checked+=len(batch)
                    angle=torch.tensor([[0.],[90.],[180.],[270.],[359.9]])
                    encoded=(torch.deg2rad(angle).cos(),torch.deg2rad(angle).sin())
                    recovered=prediction_for(name,encoded)
                    difference=abs((recovered-angle+180)%360-180)
                    assert difference.max()<1e-3
                    print('BAZ short-window audit:',checked,'real records across all splits; cos/sin encoding verified',flush=True)
        print(name,'parameters=',sum(p.numel() for p in model.parameters()),
              'trainable=',sum(p.numel() for p in model.parameters() if p.requires_grad),
              'forward/backward/zero-input checks PASS',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true',help='Check only, no training/data bank loading')
    parser.add_argument('--new-run',action='store_true')
    parser.add_argument('--config',type=Path,default=HERE/'train_config.json')
    args=parser.parse_args();cfg=json.loads(args.config.read_text(encoding='utf-8'))
    if not cfg['models'] or not set(cfg['models'])<= {'distpt_distance_only','baz_network'}:raise ValueError('Invalid models')
    if len(set(cfg['models']))!=len(cfg['models']):raise ValueError('Duplicate models')
    if any(cfg[k]<1 for k in ['batch_size','epochs','patience','cpu_threads','h5_read_processes']):raise ValueError('Invalid config')
    torch.set_num_threads(cfg['cpu_threads']);torch.set_num_interop_threads(1)
    directory=resolve_dataset(cfg['dataset_dir'])
    if args.check:preflight(cfg,directory);return
    if cfg['device'].startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no automatic CPU fallback')
    identity=dict(config=cfg,dataset=str(directory),cohort_sha256=shared.sha(directory/'cohort.npz'),
        dataset_manifest_sha256=shared.sha(directory/'manifest.json'),code_sha256={str(p):shared.sha(p) for p in
          [Path(__file__),HERE/'models.py',HERE.parent/'train_comparison.py',HERE.parent/'comparison_dataset.py',*shared.UPSTREAM.glob('*.py')]})
    results=HERE/'training_results';results.mkdir(exist_ok=True)
    active=results/'active_run.json'
    if active.exists() and not args.new_run:
        run=Path(json.loads(active.read_text())['directory'])
        if json.loads((run/'identity.json').read_text())!=identity:raise RuntimeError('Code/config/data changed; use --new-run to keep old results intact')
    else:
        preflight(cfg,directory)
        run=results/datetime.now().strftime('%Y%m%d_%H%M%S_%f');run.mkdir()
        (run/'identity.json').write_text(json.dumps(identity,indent=2))
        active.write_text(json.dumps(dict(directory=str(run)),indent=2))
    import msvcrt
    with (run/'run.lock').open('a+b') as lock:
        lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        pending=[n for n in cfg['models'] if not (run/n/'completed.json').exists()]
        if not pending:print('All additional models already completed:',run);return
        logging.basicConfig(force=True,level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',
            handlers=[logging.StreamHandler(),logging.FileHandler(run/'experiment.log',encoding='utf-8')])
        shared.make_model=create_model;shared.task_for=task_for;shared.loss_for=loss_for;shared.epoch_pass=epoch_pass
        shared.MEMORY_BANK=load_memory_bank(directory,processes=cfg['h5_read_processes'],audit_path=run/'split_audit.json')
        for name in cfg['models']:
            if name not in pending:print('Skip completed:',name);continue
            try:
                shared.train_model(name,cfg,directory,run)
            except Exception:
                logging.exception('Training stopped; completed epochs/checkpoints retained. Restart resumes this model.')
                raise
            if torch.cuda.is_available():torch.cuda.empty_cache()
        summary={name:json.loads((run/name/'completed.json').read_text()) for name in cfg['models']}
        (run/'summary.json').write_text(json.dumps(summary,indent=2))
    print('Completed:',run,flush=True)


if __name__=='__main__':
    mp.freeze_support();main()
