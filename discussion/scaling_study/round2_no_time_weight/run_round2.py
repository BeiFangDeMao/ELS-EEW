"""Independent round-two entry. No round-one pointer/results are written."""
import argparse
from datetime import datetime
import gc
import hashlib
import json
import logging
import multiprocessing
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path.insert(0, str(PARENT))


def configuration(path):
    import yaml
    import study
    settings = yaml.safe_load(path.read_text(encoding='utf-8'))
    if settings['models'] != ['ELS-2K', 'ELS-4K', 'ELS-16K', 'ELS-32K']:
        raise ValueError('Round two is restricted to 2K, 4K, 16K and 32K, in that order')
    if settings['time_weighting'] != 'none' or settings['time_weight_value'] != 1.0:
        raise ValueError('Removing temporal weights means every time weight is exactly 1')
    cfg = study.setup((path.parent / settings['base_config']).resolve())
    cfg.update(time_weighting='none', time_weight_value=1.0,
               models=settings['models'], experiment_round=2)
    cfg['results'] = str((study.WORKSPACE / settings['results']).resolve())
    cfg['round_one_results'] = str((study.WORKSPACE / settings['round_one_results']).resolve())
    return cfg


def summarize(folder, cfg):
    """Statistics only. Do not rewrite any paper figures or round-one outputs."""
    import pandas as pd
    from training_adapter import atomic_json
    from reporting import pareto_tables, recommendations
    accuracy, states, training = [], [], []
    for name in cfg['models']:
        model_dir = folder / name
        state_file = model_dir / 'state.json'
        state = json.loads(state_file.read_text(encoding='utf-8')) if state_file.exists() else {'status': 'pending'}
        states.append(dict(model=name, status=state['status'], error=state.get('error')))
        if state['status'] == 'complete':
            accuracy.append(pd.read_csv(model_dir / 'accuracy_by_time.csv'))
            training.append(json.loads((model_dir / 'training_summary.json').read_text(encoding='utf-8')))
    atomic_json(folder / 'run_status.json', states)
    pd.DataFrame(training).to_csv(folder / 'training_summary.csv', index=False)
    lines = ['# 规模实验第二轮：取消时间权重',
             '全部93个时刻权重为1，保留可学习任务权重。四个规模从头训练；8K不参与本轮。',
             '其余数据、初始化种子、优化器、学习率、梯度裁剪和至少两任务改善的检查点选择规则沿用第一轮。',
             '单随机种子42，波形记录级评价；测试集不用于选择检查点。',
             '3/5/10/20/29.76秒分别取2.88/4.80/9.92/19.84/29.76秒输出。',
             f"已完成 {len(accuracy)}/4。", *[f"- {s['model']}: {s['status']}" for s in states]]
    if accuracy:
        frame = pd.concat(accuracy, ignore_index=True)
        frame.to_csv(folder / 'accuracy_by_time.csv', index=False)
        resources = pd.read_csv(folder / 'compute_resources.csv')
        pareto_tables(frame, resources).to_csv(folder / 'pareto_results.csv', index=False)
        recommendations(frame, resources, cfg).to_csv(folder / 'validation_budget_recommendations.csv', index=False)
        old_path = Path(cfg['round_one_results']) / 'accuracy_by_time.csv'
        if old_path.exists():
            old = pd.read_csv(old_path)
            keys = ['model', 'split', 'requested_time_s', 'actual_output_time_s', 'output_chunk', 'records']
            compared = frame.merge(old, on=keys, suffixes=('_round2', '_round1'), validate='one_to_one')
            if len(compared) != len(frame):
                raise RuntimeError('Round-one/round-two sample/time alignment mismatch')
            for metric in ['magnitude_mae', 'distance_mae_km', 'azimuth_circular_mae_deg']:
                compared[metric + '_delta_round2_minus_round1'] = compared[metric + '_round2'] - compared[metric + '_round1']
            compared.to_csv(folder / 'round1_vs_round2.csv', index=False)
        lines += ['\n## 测试指标', frame[frame.split == 'test'].to_csv(index=False)]
    (folder / 'summary.md').write_text('\n\n'.join(lines), encoding='utf-8')


def execute(config_path, check=False, new_run=False):
    import pandas as pd
    import torch
    import study
    from training_adapter import (atomic_json, configure_original, seed_all, train_model,
                                  inference_metrics, original)
    from model_scaling import make_model, validate_streaming
    from unweighted_loss import equal_time_loss
    cfg = configuration(config_path)
    checks = HERE / 'checks'
    checks.mkdir(exist_ok=True)
    study.set_logging(checks / 'preflight.log')
    torch.set_num_threads(4)
    specs, resources, base_fingerprint, report = study.preflight(cfg, checks)
    specs = [next(s for s in specs if s['model'] == name) for name in cfg['models']]
    resources = [r for r in resources if r['model'] in cfg['models']]
    sources = {p.name: study.sha256(p) for p in sorted(HERE.glob('*.py'))}
    identity = dict(config=cfg, base_fingerprint=base_fingerprint, sources=sources)
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    # Full real-data shape/streaming checks above; tiny loss and resume checks below.
    from verify_round2 import verify
    with equal_time_loss(original):
        verification = verify(cfg, checks)
    atomic_json(checks / 'round2_verification.json', verification)
    if check:
        logging.info('ROUND TWO CHECK PASSED. No real-data training started; round one unchanged.')
        return
    if cfg['require_cuda_for_training'] and not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no silent CPU fallback')
    root = Path(cfg['results'])
    root.mkdir(parents=True, exist_ok=True)
    pointer = HERE / 'active_run.json'
    if pointer.exists() and not new_run:
        name = json.loads(pointer.read_text(encoding='utf-8'))['run_directory_name']
        if Path(name).name != name or '/' in name or '\\' in name:
            raise ValueError('Invalid run directory')
    else:
        name = datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_seed42_no_time_weight'
    folder = root / name
    folder.mkdir(exist_ok=True)
    with study.run_lock(folder):
        identity_file = folder / 'experiment_identity.json'
        if identity_file.exists():
            if json.loads(identity_file.read_text(encoding='utf-8'))['fingerprint'] != fingerprint:
                raise RuntimeError('Round-two source/config changed; use --new-run rather than mixing checkpoints')
        else:
            atomic_json(identity_file, dict(fingerprint=fingerprint, **identity, preflight=report))
        atomic_json(pointer, dict(run_directory_name=name, results_root=str(root)))
        study.set_logging(folder / 'experiment.log')
        logging.info('ROUND TWO: 2K -> 4K -> 16K -> 32K, all temporal weights=1.0; learnable task weights retained; 8K excluded.')
        atomic_json(folder / 'verification.json', verification)
        pd.DataFrame(resources).to_csv(folder / 'compute_resources.csv', index=False)
        atomic_json(folder / 'model_configs.json', specs)
        atomic_json(folder / 'fairness_audit.json', report['fairness'])
        pending = []
        for spec in specs:
            model_dir = folder / spec['model']
            state_file = model_dir / 'state.json'
            state = json.loads(state_file.read_text(encoding='utf-8')) if state_file.exists() else {}
            if state.get('fingerprint') not in (None, fingerprint):
                raise RuntimeError('Model fingerprint mismatch')
            if state.get('status') == 'complete':
                hashes = state.get('output_hashes', {})
                if not hashes or any(not (model_dir / p).exists() or study.sha256(model_dir / p) != h for p, h in hashes.items()):
                    raise RuntimeError('Completed output changed; refusing to overwrite')
                logging.info('SKIP completed %s', spec['model'])
            else:
                pending.append(spec)
        summarize(folder, cfg)
        if not pending:
            return
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        configure_original(cfg, device)
        splits = study.load_splits(cfg, folder)
        with equal_time_loss(original):
            for spec in pending:
                model_dir = folder / spec['model']
                model_dir.mkdir(exist_ok=True)
                handler = logging.FileHandler(model_dir / 'training.log', encoding='utf-8')
                handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
                logging.getLogger().addHandler(handler)
                try:
                    atomic_json(model_dir / 'config.json', dict(model=spec, experiment=cfg, fingerprint=fingerprint))
                    atomic_json(model_dir / 'state.json', dict(status='running', fingerprint=fingerprint))
                    seed_all(cfg['seed'])
                    model = make_model(spec)
                    loaders = study.loaders_for(splits, cfg)
                    info = study.completed_training(model, model_dir, cfg, fingerprint, device)
                    if info is None:
                        info = train_model(model, spec, cfg, (loaders['train'], loaders['validation']), model_dir, fingerprint)
                    stream_check = validate_streaming(model, torch.zeros(1, 3, 2976, device=device))
                    atomic_json(model_dir / 'streaming_validation.json', stream_check)
                    metrics = inference_metrics(model, loaders['validation'], cfg, 'validation')
                    metrics += inference_metrics(model, loaders['test'], cfg, 'test')
                    for row in metrics:
                        row.update(model=spec['model'], time_weighting='none')
                    pd.DataFrame(metrics).to_csv(model_dir / 'accuracy_by_time.csv', index=False)
                    atomic_json(model_dir / 'training_summary.json', dict(model=spec['model'], time_weighting='none', **info))
                    outputs = ['config.json', 'accuracy_by_time.csv', 'training_summary.json', 'streaming_validation.json',
                               'best.pth', 'resume_checkpoint.pth', 'training_history.csv', 'training_history.json']
                    atomic_json(model_dir / 'state.json', dict(status='complete', fingerprint=fingerprint,
                                output_hashes={p: study.sha256(model_dir / p) for p in outputs}))
                    summarize(folder, cfg)
                    del model, loaders
                    gc.collect()
                    torch.cuda.empty_cache()
                except BaseException as exc:
                    logging.exception('Round two failed; committed epoch checkpoint preserved')
                    atomic_json(model_dir / 'state.json', dict(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                                fingerprint=fingerprint, error=str(exc)))
                    summarize(folder, cfg)
                    raise
                finally:
                    logging.getLogger().removeHandler(handler)
                    handler.close()
        logging.info('All four round-two experiments complete. Results: %s', folder)


def main():
    parser = argparse.ArgumentParser(description='Round two: equal temporal weights, four scales only')
    parser.add_argument('--config', type=Path, default=HERE / 'experiment_config.yaml')
    parser.add_argument('--check', action='store_true', help='Small real-data preflight and synthetic checks only')
    parser.add_argument('--new-run', action='store_true', help='New independent round-two run; default resumes this round only')
    args = parser.parse_args()
    execute(args.config.resolve(), check=args.check, new_run=args.new_run)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
