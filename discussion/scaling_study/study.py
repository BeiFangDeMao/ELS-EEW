"""Unified preflight, auditing, sequential execution and resume control."""
import contextlib
from datetime import datetime
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import sys
import traceback
import numpy as np
import pandas as pd
import torch
import yaml

HERE=Path(__file__).resolve().parent
WORKSPACE=HERE.parents[1]


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def setup(cfg_path):
    cfg=yaml.safe_load(Path(cfg_path).read_text(encoding='utf-8'))
    for key in ['project','dataset','reference_experiment','reference_checkpoint','label_cache','data_provenance','results']:
        cfg[key]=str((WORKSPACE/cfg[key]).resolve())
    for key in ['dataset','reference_checkpoint','label_cache','data_provenance']:
        if not Path(cfg[key]).is_file(): raise FileNotFoundError(f'{key}: {cfg[key]}')
    expected={'seed':42,'batch_size':512,'epochs':300,'patience':20,'grad_clip':.8,
              'sequence_length':2976,'sampling_rate':100,'chunk_size':32,'cycle_up_epochs':10,'cycle_down_epochs':10}
    for key,value in expected.items():
        if cfg[key]!=value: raise ValueError(f'This fixed study requires {key}={value}, found {cfg[key]}')
    sys.path.insert(0,cfg['project'])
    return cfg


def set_logging(path):
    logging.basicConfig(level=logging.INFO,force=True,format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.StreamHandler(),logging.FileHandler(path,encoding='utf-8')])
    logging.getLogger('fontTools').setLevel(logging.WARNING)


def resume_fingerprint(folder,cfg,runtime_fingerprint):
    """Only an explicitly audited runtime-only repair may reuse old identity.

    Never rewrite old checkpoints or weaken training/config/source matching.
    The repair receipt pins both full runtime identity and old identity bytes.
    """
    identity_path=folder/'experiment_identity.json'
    if not identity_path.exists(): return runtime_fingerprint
    identity=json.loads(identity_path.read_text(encoding='utf-8'))
    original=identity['fingerprint']
    if original==runtime_fingerprint: return original
    receipt=folder/'runtime_repair_compatibility.json'
    if receipt.exists():
        repair=json.loads(receipt.read_text(encoding='utf-8'))
        if (repair.get('training_fingerprint')==original and
                repair.get('runtime_fingerprint')==runtime_fingerprint and
                repair.get('original_identity_sha256')==sha256(identity_path) and
                identity['config']==cfg and repair.get('config')==cfg):
            logging.info('Using audited runtime-only repair; original training identity and checkpoints preserved')
            return original
    raise RuntimeError('Config/source/data changed since previous run without an audited compatible repair; choose a new results path')


def completed_training(model,folder,cfg,fingerprint,device):
    """Post-training checks may fail after the terminal epoch was committed.

    Resume evaluation directly without touching trained checkpoint files or
    constructing a fresh training cycle. In-progress checkpoints still use the
    full optimizer/scheduler/RNG resume path in training_adapter.
    """
    path=folder/'resume_checkpoint.pth'
    if not path.exists(): return None
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state['fingerprint']!=fingerprint: raise RuntimeError('Resume fingerprint mismatch')
    if state['epoch']<cfg['epochs'] and state['bad_epochs']<cfg['patience']: return None
    best=state['best_payload']
    if best is None: raise RuntimeError('Terminal training checkpoint has no best model')
    model.load_state_dict(best['model_state_dict'],strict=True)
    model.to(device)
    logging.info('Training already complete at epoch %d (best %d); skip training and resume evaluation only',
                 state['epoch'],state['best_epoch'])
    return dict(status='training_complete',completed_epochs=len(state['history']),best_epoch=state['best_epoch'],
                historical_bests=state['historical_bests'],best_validation=best['validation_metrics'])


@contextlib.contextmanager
def run_lock(folder):
    """OS lock released even after crashes. Persistent file is not a stale PID lock."""
    path=folder/'run.lock'
    stream=path.open('a+b')
    stream.seek(0)
    if not stream.read(1): stream.write(b'0');stream.flush()
    stream.seek(0)
    try:
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError as exc:
        stream.close()
        raise RuntimeError('Another scaling process holds this results directory') from exc
    try: yield
    finally: stream.close()


def audit_reference(cfg):
    from training import Trainstudents as original
    from model_scaling import make_model
    checkpoint=torch.load(cfg['reference_checkpoint'],map_location='cpu',weights_only=False)
    run=json.loads((Path(cfg['reference_experiment'])/'run_config.json').read_text(encoding='utf-8'))
    archive=Path(cfg['reference_checkpoint']).parent/'test_metrics_by_chunk.json'
    metadata=json.loads(archive.read_text(encoding='utf-8'))['metadata']
    mismatches=[]
    for key in ['batch_size','epochs','patience','grad_clip','exclude_200hz','sequence_length','seed']:
        if cfg[key]!=run[key]: mismatches.append(f'{key}: reference={run[key]}, study={cfg[key]}')
    if Path(run['train_h5']).name != Path(cfg['dataset']).name:
        mismatches.append('Reference and study dataset filenames differ')
    arch=checkpoint.get('architecture_config')
    if arch != dict(width=16,hidden=24,kind='ds',dilations=(1,2,4,8),mode='gru'):
        mismatches.append(f'Unexpected reference architecture: {arch}')
    if sha256(cfg['reference_checkpoint'])!=cfg['reference_sha256']:
        mismatches.append('Reference weight SHA256 differs from the designated experiment')
    for key,expected in [('sequence_length',2976),('num_chunks',93),('grad_clip',.8),('num_parameters',8263)]:
        if checkpoint.get(key)!=expected: mismatches.append(f'Checkpoint {key} differs: {checkpoint.get(key)}')
    groups=checkpoint['optimizer_state_dict']['param_groups']
    for group,base in zip(groups,[cfg['backbone_base_lr'],cfg['head_base_lr']]):
        if group['weight_decay']!=cfg['weight_decay'] or group['initial_lr']!=base or tuple(group['betas'])!=(.9,.999) or group['eps']!=1e-8:
            mismatches.append('Archived optimizer settings differ')
    from training_adapter import configure_original
    configure_original(cfg,torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    provenance=json.loads(Path(cfg['data_provenance']).read_text(encoding='utf-8'))
    stat=Path(cfg['dataset']).stat()
    if stat.st_size!=provenance['source_size'] or stat.st_mtime_ns!=provenance['source_mtime_ns']:
        mismatches.append('Dataset size/mtime differs from original selection provenance; cached splits not trusted')
    if provenance['counts']!=cfg['expected_split_counts']:
        mismatches.append('Expected split counts differ from original provenance')
    model=make_model(dict(cnn_width=16,gru_hidden=24))
    model.load_state_dict(checkpoint['model_state_dict'],strict=True)
    report=dict(reference_parameters=sum(p.numel() for p in model.parameters()),reference_epoch=checkpoint['epoch'],
                reference_completed_epochs=metadata['completed_epochs'],architecture=arch,mismatches=mismatches,
                prior_test_exposure=True,
                warnings=[
                    'Original test results have already been inspected during prior design; this is not an untouched development test.',
                    'No original training-source snapshot exists in this checkpoint. Saved run config, optimizer, logs and present implementation are audited; historical source identity cannot be proven.',
                    'Original epoch code skips invalid batches. This wrapper aborts on the first skipped-batch warning. Finite-batch math is unchanged.',
                    'Original test code substitutes NaN/Inf; this study aborts instead. Valid-output clipping is identical.',
                    'Single-seed study; candidate-set empirical Pareto fronts do not establish statistical superiority.'
                ],decision='reuse_existing_8K' if not cfg['retrain_8k'] else 'explicitly_retrain_8K')
    if mismatches and not cfg['retrain_8k'] and not cfg['allow_reference_mismatch']:
        report['decision']='BLOCKED: reference mismatch; inspect report or explicitly enable retrain_8k'
    return report,checkpoint,model


def preflight(cfg,folder):
    from model_scaling import generate_configs,make_model,resources,validate_streaming
    from training_adapter import atomic_json,seed_all
    folder.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(WORKSPACE/'plotting'))
    from publication_style import setup as plot_style
    plot_fonts=plot_style('zh')[2]
    plot_style('en')
    report,reference,reference_model=audit_reference(cfg)
    atomic_json(folder/'fairness_audit.json',report)
    if report['mismatches'] and not cfg['retrain_8k'] and not cfg['allow_reference_mismatch']:
        raise RuntimeError('Reference audit failed: '+ '; '.join(report['mismatches']))
    specs,candidates=generate_configs()
    pd.DataFrame(candidates).to_csv(folder/'configuration_candidates.csv',index=False)
    import h5py
    # Read only six real records. This is NOT full inference or dataset loading.
    with np.load(cfg['label_cache'],allow_pickle=False) as cache:
        keys=np.asarray(cache['keys'])
        split=np.asarray(cache['splits'])
        assert {s:int((split==s).sum()) for s in ('train','validation','test')}==cfg['expected_split_counts']
        selected=[str(keys[np.flatnonzero(split==s)[i]]) for s in ('train','validation','test') for i in (0,1)]
    with h5py.File(cfg['dataset'],'r') as file:
        samples=[]
        for key in selected:
            ds=file[key]
            wave=ds[:].astype(np.float32).T
            assert wave.shape==(3,2976) and np.isfinite(wave).all()
            assert ds.attrs['sampling_freq_hz']==100
            assert str(ds.attrs['channel_order'])=='Z,N,E' and str(ds.attrs['waveform_unit'])=='gal'
            samples.append(wave)
        root_processing={str(k):str(v) for k,v in file.attrs.items()}
    wave=torch.from_numpy(np.stack(samples[:2]))
    seed_all(cfg['seed'])
    rows,checks=[],{}
    for spec in specs:
        model=reference_model if spec['model']=='ELS-8K' else make_model(spec)
        row=resources(model,spec)
        rows.append(row)
        checks[spec['model']]=validate_streaming(model,wave)
        logging.info('%s: CNN=%d, GRU hidden=%d x2, parameters=%d, MACs/update=%d, state=%.4f KiB',
            spec['model'],spec['cnn_width'],spec['gru_hidden'],row['total_parameters'],row['macs_per_update'],row['persistent_state_kib'])
    configs=pd.DataFrame(specs).merge(pd.DataFrame(rows),on=['model','cnn_width','gru_hidden','gru_layers','total_parameters'])
    configs.to_csv(folder/'model_configs.csv',index=False)
    pd.DataFrame(rows).to_csv(folder/'compute_resources.csv',index=False)
    atomic_json(folder/'model_configs.json',specs)
    # Capture all executable sources that can alter model/training/statistics.
    source_files=[Path(cfg['project'])/p for p in ['models/compact_gru.py','models/common.py','models/students.py',
        'training/Trainstudents.py','training/h5_loading.py','training/causal_preprocessing.py']]
    source_files+=list(HERE.glob('*.py'))
    source_hashes={str(p.relative_to(WORKSPACE)):sha256(p) for p in source_files}
    identity=dict(config=cfg,models=specs,sources=source_hashes,
                  reference_sha256=sha256(cfg['reference_checkpoint']),dataset_stat=[Path(cfg['dataset']).stat().st_size,Path(cfg['dataset']).stat().st_mtime_ns],
                  selection_cache_sha256=sha256(cfg['label_cache']))
    fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    result=dict(passed=True,scope='six real records + five full 2976-point forward/stream checks; no formal training',
                torch=torch.__version__,python=sys.version,platform=platform.platform(),cuda=torch.cuda.is_available(),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                configuration_count=len(specs),inference_checks=checks,data_root_metadata=root_processing,
                original_weights_sha256=sha256(cfg['reference_checkpoint']),fingerprint=fingerprint,
                source_hashes=source_hashes,fairness=report)
    result['plot_fonts']=plot_fonts
    atomic_json(folder/'preflight_report.json',result)
    return specs,rows,fingerprint,result


def load_splits(cfg,folder):
    from training import Trainstudents as original
    data=original.SeismicDataset(cfg['dataset'],require_year=True,exclude_200hz=cfg['exclude_200hz'],read_processes=cfg['h5_read_processes'])
    splits=dict(train=[],validation=[],test=[])
    for item in data.data_list:
        year=item[3]
        split='validation' if year in ['19','20','21','22'] else 'test' if year in ['23','24'] else 'train'
        splits[split].append(item)
    with np.load(cfg['label_cache'],allow_pickle=False) as cache:
        keys,saved_split=cache['keys'],cache['splits']
        for name,items in splits.items():
            actual=[item[2] for item in items]
            expected=keys[saved_split==name].tolist()
            if actual!=expected: raise RuntimeError(f'{name} ordered record list differs from original selection')
            if len(actual)!=cfg['expected_split_counts'][name]: raise RuntimeError(f'Bad {name} count')
    from training_adapter import atomic_json
    atomic_json(folder/'split_audit.json',{name:dict(records=len(items),ordered_keys_sha256=hashlib.sha256('\n'.join(x[2] for x in items).encode()).hexdigest()) for name,items in splits.items()})
    del data
    return splits


def loaders_for(splits,cfg):
    from torch.utils.data import DataLoader
    from training import Trainstudents as original
    return {name:DataLoader(items,batch_size=cfg['batch_size'],shuffle=name=='train',collate_fn=original.collate_fn,num_workers=0)
            for name,items in splits.items()}


def execute(config_path,check_only=False,new_run=False):
    # Check runs cannot create completion states or formal checkpoints.
    pre=HERE/'checks'
    pre.mkdir(parents=True,exist_ok=True)
    set_logging(pre/'preflight.log')
    cfg=setup(config_path)
    torch.set_num_threads(4)
    specs,resource_rows,fingerprint,report=preflight(cfg,pre)
    if check_only:
        from synthetic_checks import run_checks
        from training_adapter import atomic_json
        checks=run_checks(cfg)
        atomic_json(pre/'synthetic_checks.json',checks)
        logging.info('PRECHECK PASSED. No complete training started. Formal results untouched.')
        return
    if cfg['require_cuda_for_training'] and not torch.cuda.is_available():
        raise RuntimeError('CUDA required for formal training; no automatic CPU fallback')
    from training_adapter import atomic_json,configure_original,seed_all,train_model,inference_metrics
    from model_scaling import make_model,validate_streaming
    from reporting import summarize
    root=Path(cfg['results'])
    pointer=HERE/'active_run.json'
    if pointer.exists() and not new_run:
        name=json.loads(pointer.read_text(encoding='utf-8'))['run_directory_name']
        if Path(name).name!=name or '/' in name or '\\' in name:
            raise RuntimeError('Invalid active run directory name')
    else:
        name=datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_seed42'
    folder=root/name
    folder.mkdir(parents=True,exist_ok=True)
    with run_lock(folder):
        atomic_json(pointer,dict(run_directory_name=name,results_root=str(root)))
        set_logging(folder/'experiment.log')
        existing=folder/'experiment_identity.json'
        runtime_fingerprint=fingerprint
        fingerprint=resume_fingerprint(folder,cfg,runtime_fingerprint)
        if not existing.exists():
            atomic_json(existing,dict(fingerprint=fingerprint,config=cfg,preflight=report))
        atomic_json(folder/'current_runtime_report.json',dict(runtime_fingerprint=runtime_fingerprint,
            training_fingerprint=fingerprint,preflight=report))
        pd.DataFrame(specs).merge(pd.DataFrame(resource_rows),on=['model','cnn_width','gru_hidden','gru_layers','total_parameters']).to_csv(folder/'model_configs.csv',index=False)
        pd.DataFrame(resource_rows).to_csv(folder/'compute_resources.csv',index=False)
        atomic_json(folder/'fairness_audit.json',report['fairness'])
        logging.info('Plan: reference8K %s; sequential 2K -> 4K -> 16K -> 32K; max 300 epochs each; full fixed splits; no simultaneous training.',
                     'retrain explicitly enabled' if cfg['retrain_8k'] else 'load existing weights')
        logging.info('Expected counts %s. Source loader eagerly holds ~26.5 GB waveform data plus metric arrays. No automatic batch-size reduction.',cfg['expected_split_counts'])
        pending=[]
        # Reference is evaluated first, then four trained models.
        order=sorted(specs,key=lambda s:(s['model']!='ELS-8K',s['target_parameters']))
        for spec in order:
            state_path=folder/spec['model']/'state.json'
            state=json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {}
            if state.get('fingerprint') not in (None,fingerprint): raise RuntimeError('Model state fingerprint differs')
            if state.get('status')=='complete':
                outputs=state.get('output_hashes',{})
                if not outputs or any(not (state_path.parent/p).is_file() or sha256(state_path.parent/p)!=h for p,h in outputs.items()):
                    raise RuntimeError(f'Completed outputs missing/changed: {spec["model"]}; inspect rather than silently overwriting')
                logging.info('SKIP completed model %s',spec['model'])
            else: pending.append(spec)
        if not pending:
            summarize(folder,cfg)
            return
        configure_original(cfg,torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        splits=load_splits(cfg,folder)
        for n,spec in enumerate(pending,1):
            model_folder=folder/spec['model']
            model_folder.mkdir(exist_ok=True)
            atomic_json(model_folder/'config.json',dict(model=spec,experiment=cfg,fingerprint=fingerprint))
            handler=logging.FileHandler(model_folder/'training.log',encoding='utf-8')
            handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
            logging.getLogger().addHandler(handler)
            try:
                logging.info('MODEL %d/%d pending: %s',n,len(pending),spec['model'])
                atomic_json(model_folder/'state.json',dict(status='running',fingerprint=fingerprint))
                seed_all(cfg['seed'])
                model=make_model(spec)
                loaders=loaders_for(splits,cfg)
                if spec['model']=='ELS-8K' and not cfg['retrain_8k']:
                    checkpoint=torch.load(cfg['reference_checkpoint'],map_location='cpu',weights_only=False)
                    model.load_state_dict(checkpoint['model_state_dict'],strict=True)
                    model.to('cuda' if torch.cuda.is_available() else 'cpu')
                    info=dict(status='reference_reused',best_epoch=checkpoint['epoch'],completed_epochs=report['fairness']['reference_completed_epochs'],
                              weight_source=cfg['reference_checkpoint'],reference_retrained=False)
                else:
                    info=completed_training(model,model_folder,cfg,fingerprint,
                        torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                    if info is None:
                        info=train_model(model,spec,cfg,(loaders['train'],loaders['validation']),model_folder,fingerprint)
                stream_check=validate_streaming(model,torch.zeros(1,3,2976,device=next(model.parameters()).device))
                atomic_json(model_folder/'streaming_validation.json',stream_check)
                # Validation is processed first and is the ONLY split used by
                # resource-budget recommendations / model-size selection.
                metrics=inference_metrics(model,loaders['validation'],cfg,'validation')
                metrics+=inference_metrics(model,loaders['test'],cfg,'test')
                for row in metrics: row['model']=spec['model']
                pd.DataFrame(metrics).to_csv(model_folder/'accuracy_by_time.csv',index=False)
                atomic_json(model_folder/'training_summary.json',dict(model=spec['model'],**info))
                atomic_json(model_folder/'compute_resources.json',next(r for r in resource_rows if r['model']==spec['model']))
                output_files=['accuracy_by_time.csv','training_summary.json','compute_resources.json','config.json']
                if (model_folder/'best.pth').exists(): output_files+=['best.pth','resume_checkpoint.pth']
                atomic_json(model_folder/'state.json',dict(status='complete',fingerprint=fingerprint,
                    output_hashes={name:sha256(model_folder/name) for name in output_files},completed_at=datetime.now().isoformat()))
                summarize(folder,cfg)
                del model,loaders
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
            except BaseException as exc:
                logging.exception('%s failed; committed epoch checkpoints are preserved. No settings will be reduced.',spec['model'])
                atomic_json(model_folder/'state.json',dict(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
                    fingerprint=fingerprint,error=str(exc),traceback=traceback.format_exc()))
                summarize(folder,cfg)
                raise
            finally:
                logging.getLogger().removeHandler(handler)
                handler.close()
        if sha256(cfg['reference_checkpoint'])!=report['original_weights_sha256']:
            raise RuntimeError('Original reference weight changed externally during the experiment')
        logging.info('All five scaling experiments complete. Results: %s',folder)
