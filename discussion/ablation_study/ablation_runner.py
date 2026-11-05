"""Independent 8K-only ablation orchestration; never edits prior experiments."""
from datetime import datetime
import gc
import hashlib
import json
import logging
from pathlib import Path
import sys
import traceback
import numpy as np
import pandas as pd
import torch
import yaml

HERE=Path(__file__).resolve().parent
WORKSPACE=HERE.parents[1]


def configure(path):
    extra=yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    sys.path.insert(0,str(WORKSPACE/'discussion/scaling_study'))
    import study
    base=study.setup(WORKSPACE/extra['scaling_config'])
    for key in ['project','reference_checkpoint','pure_checkpoint','pure_run_config','pure_metrics','results']:
        extra[key]=str((WORKSPACE/extra[key]).resolve())
    cfg={**base,**extra}
    sys.path.insert(0,cfg['project'])
    sys.path.insert(0,str(WORKSPACE/'discussion/early_reliability_warning'))
    if not cfg['reuse_8k'] or not cfg['reuse_pure_gru']: raise ValueError('Existing 8K/PureGRU must be reused, not trained')
    if cfg['parameter_matched_pure_policy']!='report_unavailable_do_not_train': raise ValueError('No new PureGRU training or implementation allowed')
    if set(cfg['groups'])!={'architecture','multitask','loss','cache'}: raise ValueError('Unknown/missing group switch')
    for key in ['records','repeats','threads']:
        if cfg['cache'][key]<1: raise ValueError(f'cache.{key} must be positive')
    if cfg['cache']['end_to_end_preprocessing']:
        raise ValueError('H5 is already preprocessed. Do not filter it again; optional raw end-to-end benchmark is not enabled here.')
    return cfg


def load_existing(spec,cfg):
    from ablation_models import build_model
    path=cfg['pure_checkpoint'] if spec['reuse']=='pure' else cfg['reference_checkpoint']
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    model=build_model(spec)
    model.load_state_dict(checkpoint['model_state_dict'],strict=True)
    return model,checkpoint,path


def preflight(cfg,folder):
    from study import audit_reference,sha256
    from training_adapter import atomic_json
    from ablation_models import experiments,build_model,resource_stats
    from model_scaling import validate_streaming
    folder.mkdir(parents=True,exist_ok=True)
    audit,_,_=audit_reference(cfg)
    if audit['mismatches']: raise RuntimeError('8K configuration mismatch: '+str(audit['mismatches']))
    pure=None
    if cfg['groups']['architecture']:
        pure=json.loads(Path(cfg['pure_run_config']).read_text(encoding='utf-8'))
        differences=[]
        for key in ['sequence_length','batch_size','epochs','patience','grad_clip','exclude_200hz','seed']:
            if pure[key]!=cfg[key]: differences.append(dict(field=key,old=pure[key],new=cfg[key]))
        if Path(pure['train_h5']).name!=Path(cfg['dataset']).name: differences.append(dict(field='dataset',old=pure['train_h5'],new=cfg['dataset']))
        saved=json.loads(Path(cfg['pure_metrics']).read_text(encoding='utf-8'))['metadata']
        for key,expected in [('test_records',51772),('num_chunks',93),('num_parameters',5791)]:
            if saved.get(key)!=expected: differences.append(dict(field=key,old=saved.get(key),new=expected))
        audit['pure_gru']=dict(differences=differences,action='reuse weights and unified inference; never retrain',
            parameters=5791,parameter_matched_to_8263=False,checkpoint_sha256=sha256(cfg['pure_checkpoint']),
            original_best_epoch=saved['best_epoch'],original_completed_epochs=saved['completed_epochs'],
            note='Original trainer has an empty CNN group for PureGRU; effective GRU/head LR is unchanged. Historical training source snapshot not available.')
    audit['matched_pure']=dict(status='not_available',reason='Only existing hidden24 PureGRU (5791 params) found; user prohibits new PureGRU training/structure.')
    specs=experiments(cfg)
    import h5py
    with np.load(cfg['label_cache'],allow_pickle=False) as cache:
        keys=cache['keys'][cache['splits']=='validation'][:2].tolist()
    with h5py.File(cfg['dataset'],'r') as file:
        waves=torch.from_numpy(np.stack([file[key][:].astype(np.float32).T for key in keys]))
    resources=[];checks={}
    for spec in specs:
        model=load_existing(spec,cfg)[0] if spec['reuse'] else build_model(spec)
        resources.append(resource_stats(model,spec['name']))
        if spec['task']=='all': checks[spec['name']]=validate_streaming(model,waves)
        else:
            model.eval()
            with torch.inference_mode():
                full=model(waves)[0];state=None;pieces=[]
                for piece in waves.split(32,-1):
                    pred,state=model.stream_step(piece,state);pieces.append(pred)
                torch.testing.assert_close(full,torch.cat(pieces,dim=1),rtol=2e-4,atol=2e-5)
            checks[spec['name']]=dict(shape=list(full.shape),stream_match=True)
        logging.info('%s: %s; params=%d trainable=%d',spec['name'],'reuse existing' if spec['reuse'] else 'NEW TRAINING',
                     resources[-1]['total_parameters'],resources[-1]['trainable_parameters'])
    probe=build_model(specs[0])
    from ablation_models import time_weights
    probe_wave=torch.zeros(1,93,4)
    means=dict(staged=float(time_weights(probe_wave,'staged',cfg['uniform_time_weight']).mean()),uniform=cfg['uniform_time_weight'])
    pd.DataFrame(specs).to_csv(folder/'experiment_plan.csv',index=False)
    pd.DataFrame(resources).to_csv(folder/'model_resources.csv',index=False)
    atomic_json(folder/'reference_audit.json',audit)
    sources=list(HERE.glob('*.py'))+[Path(cfg['project'])/p for p in ['models/compact_gru.py','models/pure_gru.py',
        'models/common.py','models/students.py','training/Trainstudents.py','training/h5_loading.py']]
    sources+=[WORKSPACE/'discussion/scaling_study'/p for p in ['training_adapter.py','model_scaling.py','study.py']]
    hashes={str(path.relative_to(WORKSPACE)):sha256(path) for path in sources}
    signature=dict(config=cfg,sources=hashes,reference_weight=sha256(cfg['reference_checkpoint']),
        pure_weight=sha256(cfg['pure_checkpoint']) if cfg['groups']['architecture'] else None,
        dataset_stat=[Path(cfg['dataset']).stat().st_size,Path(cfg['dataset']).stat().st_mtime_ns],
        label_cache=sha256(cfg['label_cache']))
    fingerprint=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
    from matplotlib import font_manager
    for font in ['SimSun','Times New Roman']: font_manager.findfont(font_manager.FontProperties(family=font),fallback_to_default=False)
    report=dict(passed=True,new_training_models=sum(s['reuse'] is None for s in specs),scope='8K backbone only; existing PureGRU exception',
                audit=audit,inference_checks=checks,time_weight_means=means,
                time_weight_mean_note='93 outputs split as 24/23/23/23: staged mean=0.848387; uniform=0.85 is approximately, not exactly, mean matched.',
                fingerprint=fingerprint,sources=hashes,real_training_started=False)
    report['protected_weights']={cfg['reference_checkpoint']:sha256(cfg['reference_checkpoint'])}
    if cfg['groups']['architecture']: report['protected_weights'][cfg['pure_checkpoint']]=sha256(cfg['pure_checkpoint'])
    atomic_json(folder/'preflight_report.json',report)
    return specs,resources,fingerprint,report,waves


def execute(config_path,check_only=False,new_run=False,cache_only=False):
    checks=HERE/'checks';checks.mkdir(parents=True,exist_ok=True)
    # Configure logging before checking data so missing files also leave a log.
    logging.basicConfig(force=True,level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.StreamHandler(),logging.FileHandler(checks/'preflight.log',encoding='utf-8')])
    logging.getLogger('fontTools').setLevel(logging.WARNING)
    cfg=configure(config_path)
    torch.set_num_threads(4)
    specs,resources,fingerprint,report,waves=preflight(cfg,checks)
    from training_adapter import atomic_json,seed_all,configure_original
    if check_only:
        from ablation_checks import run_checks
        atomic_json(checks/'synthetic_checks.json',run_checks(cfg,specs,waves))
        logging.info('ABLATION PREFLIGHT PASSED; no formal training started')
        return
    if not cache_only and cfg['require_cuda_for_training'] and not torch.cuda.is_available(): raise RuntimeError('CUDA required for real training')
    from study import run_lock,sha256,set_logging,load_splits,loaders_for
    from ablation_models import build_model
    from ablation_training import fit
    from ablation_evaluation import evaluate_model,geometry_coordinates
    from ablation_reporting import summarize
    pointer=HERE/'active_run.json'
    name=json.loads(pointer.read_text(encoding='utf-8'))['run_name'] if pointer.exists() and not new_run else datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_seed42'
    if Path(name).name!=name or '/' in name or '\\' in name: raise ValueError('Unsafe result directory name')
    root=Path(cfg['results'])/name;root.mkdir(parents=True,exist_ok=True)
    with run_lock(root):
        set_logging(root/'ablation.log')
        identity=root/'experiment_identity.json'
        if identity.exists() and json.loads(identity.read_text(encoding='utf-8'))['fingerprint']!=fingerprint:
            saved=json.loads(identity.read_text(encoding='utf-8'))
            receipt_path=root/'gradient_repair_compatibility.json'
            receipt=json.loads(receipt_path.read_text(encoding='utf-8')) if receipt_path.exists() else {}
            if (receipt.get('training_fingerprint')!=saved['fingerprint'] or
                receipt.get('runtime_fingerprint')!=fingerprint or
                receipt.get('identity_sha256')!=sha256(identity) or saved['config']!=cfg):
                raise RuntimeError('Source/config/data changed without a matching audited repair receipt. Use --new-run.')
            atomic_json(root/'current_runtime_report.json',report)
            logging.warning('Audited gradient numerical repair active; preserving original training identity. See %s',receipt_path)
            fingerprint=saved['fingerprint']
        if not identity.exists(): atomic_json(identity,dict(fingerprint=fingerprint,config=cfg,preflight=report))
        atomic_json(pointer,dict(run_name=name,results_root=str(root.parent)))
        pd.DataFrame(specs).to_csv(root/'experiment_plan.csv',index=False)
        pd.DataFrame(resources).to_csv(root/'model_resources.csv',index=False)
        atomic_json(root/'reference_audit.json',report['audit'])
        logging.info('Plan: reuse original 8K/PureGRU weights; %d new models, sequential. Prior original experiments remain read-only.',report['new_training_models'])
        pending=[]
        for spec in specs:
            statepath=root/spec['name']/'state.json'
            state=json.loads(statepath.read_text(encoding='utf-8')) if statepath.exists() else {}
            if state and state.get('fingerprint')!=fingerprint: raise RuntimeError('Per-model identity mismatch')
            if state.get('status')=='complete':
                if any(not (statepath.parent/p).exists() or sha256(statepath.parent/p)!=h for p,h in state['files'].items()):
                    raise RuntimeError('Completed output changed/missing; refusing silent overwrite')
                logging.info('SKIP completed %s',spec['name'])
            else: pending.append(spec)
        if pending and not cache_only:
            configure_original(cfg,torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            splits=load_splits(cfg,root)
            keys=[item[2] for item in splits['test']]
            geometry=root/'test_geometry.npz'
            if geometry.exists():
                with np.load(geometry,allow_pickle=False) as data:
                    if data['keys'].tolist()!=keys: raise RuntimeError('Geometry key order changed')
                    coords=data['coordinates']
            else: coords=geometry_coordinates(cfg,keys,root)
            for spec in pending:
                folder=root/spec['name'];folder.mkdir(exist_ok=True)
                atomic_json(folder/'config.json',dict(experiment=cfg,model=spec,fingerprint=fingerprint))
                handler=logging.FileHandler(folder/'training.log',encoding='utf-8');logging.getLogger().addHandler(handler)
                try:
                    atomic_json(folder/'state.json',dict(status='running',fingerprint=fingerprint))
                    seed_all(cfg['seed']);loaders=loaders_for(splits,cfg)
                    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    if spec['reuse']:
                        model,checkpoint,weight_path=load_existing(spec,cfg)
                        archive=json.loads((Path(weight_path).parent/'test_metrics_by_chunk.json').read_text(encoding='utf-8'))['metadata']
                        training=dict(reused=True,checkpoint_source=weight_path,best_epoch=checkpoint['epoch'],
                                      completed_epochs=archive['completed_epochs'],new_training_epochs=0,
                                      note='Original training skipped; unified evaluation only')
                    else:
                        model=build_model(spec)
                        training=fit(model,spec,cfg,loaders,folder,fingerprint,device)
                    evaluate_model(model,loaders['test'],spec,cfg,folder,coords,device)
                    atomic_json(folder/'training_summary.json',training)
                    files=['config.json','metrics_by_time.csv','stability.csv','predictions.npy','truth_labels.npy','training_summary.json']
                    if not spec['reuse']: files+=['best.pth','resume_checkpoint.pth','training_history.json']
                    atomic_json(folder/'state.json',dict(status='complete',fingerprint=fingerprint,files={p:sha256(folder/p) for p in files}))
                    summarize(root,cfg)
                    del model,loaders;gc.collect()
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                except BaseException as exc:
                    logging.exception('Ablation failed; saved training checkpoints remain intact')
                    atomic_json(folder/'state.json',dict(status='failed',fingerprint=fingerprint,error=str(exc),traceback=traceback.format_exc()))
                    summarize(root,cfg)
                    raise
                finally:
                    logging.getLogger().removeHandler(handler);handler.close()
            del splits;gc.collect()
        if cfg['groups']['cache'] or cache_only:
            cache_folder=root/'cache';cache_folder.mkdir(exist_ok=True)
            statepath=cache_folder/'state.json'
            previous=json.loads(statepath.read_text(encoding='utf-8')) if statepath.exists() else {}
            if previous.get('status')=='complete':
                if previous.get('fingerprint')!=fingerprint or any(sha256(cache_folder/p)!=h for p,h in previous['files'].items()):
                    raise RuntimeError('Cache completion identity/output mismatch')
                logging.info('SKIP completed cache benchmark')
            else:
                try:
                    from inference import TestData,selected_keys
                    from cache_benchmark import benchmark
                    keys,_=selected_keys();indices=np.linspace(0,len(keys)-1,cfg['cache']['records']).astype(int)
                    data=TestData(keys[indices]);cache_waves=torch.stack([torch.from_numpy(data[i][0]) for i in range(len(indices))])
                    if data.file is not None: data.file.close()
                    model=load_existing(specs[0],cfg)[0]
                    benchmark(model,cache_waves,cfg,cache_folder)
                    files=[p.name for p in cache_folder.iterdir() if p.is_file() and p.name!='state.json']
                    atomic_json(statepath,dict(status='complete',fingerprint=fingerprint,files={p:sha256(cache_folder/p) for p in files}))
                except BaseException as exc:
                    atomic_json(statepath,dict(status='failed',fingerprint=fingerprint,error=str(exc)))
                    summarize(root,cfg);raise
        summarize(root,cfg)
        if any(sha256(path)!=h for path,h in report['protected_weights'].items()):
            raise RuntimeError('Original weight changed during run')
        logging.info('Ablation run complete: %s',root)
