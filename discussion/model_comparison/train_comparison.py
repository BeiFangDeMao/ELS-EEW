"""Train four independent baselines sequentially; check mode never trains data."""
import argparse
import csv
import hashlib
import importlib.util
import json
import logging
import multiprocessing as mp
import random
import sys
import traceback
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from comparison_dataset import ComparisonDataset, resolve_dataset, load_memory_bank, MemoryComparisonDataset

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE/'upstream/SeisT-main/models'
MEMORY_BANK = None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def upstream():
    # Unique package name prevents collision with CompactGRU's models package.
    name = '_comparison_upstream_models'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, UPSTREAM/'__init__.py',
                                                     submodule_search_locations=[str(UPSTREAM)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def make_model(name, cfg):
    module = upstream()
    if name == 'seist_s_emg':
        # Official head caps magnitude at 8; our existing labels extend to 9.
        s = module.seist
        return s.SeismogramTransformer_S(output_head=partial(s.HeadRegression,
            out_act_layer=partial(s.ScaledActivation, act_layer=nn.Sigmoid,
                                  scale_factor=cfg['magnitude_output_max'])))
    return module.create_model(name, in_channels=3)


def task_for(name):
    return 'emg' if name == 'magnet' else name.rsplit('_',1)[1]


def loss_for(name, output, target):
    # Same published loss definitions; reduce to mean per record consistently.
    if name == 'magnet':
        return upstream().MousaviLoss()(output, target) / len(target)
    return nn.functional.huber_loss(output, target, delta=1., reduction='mean')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def atomic_save(value, path):
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def load_checkpoint(path, device):
    # Checkpoints here are generated locally by this script, including RNG arrays.
    return torch.load(path, map_location=device, weights_only=False)


def loader(directory, split, name, cfg, generator=None):
    if MEMORY_BANK is None:
        raise RuntimeError('必须先加载全部数据到内存')
    dataset = MemoryComparisonDataset(MEMORY_BANK, split, task_for(name))
    args = dict(batch_size=cfg['batch_size'], shuffle=split=='train',
                num_workers=0, pin_memory=cfg['device'].startswith('cuda'),
                generator=generator, drop_last=False)
    return DataLoader(dataset, **args)


def epoch_pass(model, data, name, device, optimizer=None, clip=0.8, writer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = total_error = 0.
    count = 0
    with torch.set_grad_enabled(training):
        for wave, target, keys in tqdm(data, desc='train' if training else 'evaluate', leave=False):
            wave, target = wave.to(device), target.to(device)
            output = model(wave)
            if output.shape != (len(wave), 2 if name=='magnet' else 1):
                raise ValueError(f'模型输出形状错误：{output.shape}')
            loss = loss_for(name, output, target)
            if not torch.isfinite(output).all() or not torch.isfinite(loss):
                raise FloatingPointError('非有限预测或loss；停止，不跳过批次')
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
                optimizer.step()
            prediction = output[:, :1].detach()
            error = (prediction-target).abs()
            if task_for(name) == 'baz':
                error = torch.remainder(error, 360.)
                error = torch.minimum(error, 360.-error)
            count += len(wave)
            total_loss += loss.item()*len(wave)
            total_error += error.sum().item()
            if writer:
                for key, y, p in zip(keys, target[:,0].tolist(), prediction[:,0].tolist()):
                    writer.writerow([key, y, p])
    return dict(loss=total_loss/count, mae=total_error/count, records=count)


def preflight(cfg, directory=None):
    # Synthetic inference checks all four shapes and finite backward gradients.
    for name in cfg['models']:
        seed_all(cfg['seed'])
        model = make_model(name,cfg).cpu()
        model.train()
        x = torch.randn(2,3,288)
        target = torch.full((2,1), {'emg':4.,'dis':50.,'baz':180.}[task_for(name)])
        output = model(x)
        assert output.shape == (2,2 if name=='magnet' else 1), (name,output.shape)
        loss = loss_for(name,output,target)
        if not torch.isfinite(loss):
            raise FloatingPointError(name)
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError(f'{name}: invalid gradients')
        print(name, 'parameters=',sum(p.numel() for p in model.parameters()), '288-point check PASS', flush=True)
        if directory:
            d = ComparisonDataset(directory,'train',task_for(name))
            d[0]
            d.close()


def train_model(name, cfg, directory, run):
    out = run/name
    out.mkdir(exist_ok=True)
    if (out/'completed.json').exists():
        print(f'跳过已完成模型 {name}', flush=True)
        return
    seed_all(cfg['seed'])
    device = torch.device(cfg['device'])
    model = make_model(name,cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],
                                  weight_decay=cfg['weight_decay'])
    generator = torch.Generator().manual_seed(cfg['seed'])
    start, best, bad = 0, float('inf'), 0
    resume = out/'resume.pth'
    if resume.exists():
        c = load_checkpoint(resume,'cpu')
        model.load_state_dict(c['model'])
        optimizer.load_state_dict(c['optimizer'])
        start, best, bad = c['epoch']+1, c['best'], c['bad']
        random.setstate(c['python_rng']); np.random.set_state(c['numpy_rng'])
        torch.set_rng_state(c['torch_rng'])
        if device.type=='cuda': torch.cuda.set_rng_state_all(c['cuda_rng'])
        generator.set_state(c['loader_rng'])
    train = loader(directory,'train',name,cfg,generator)
    validation = loader(directory,'validation',name,cfg)
    handler = logging.FileHandler(out/'training.log',encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    log = logging.getLogger(); log.setLevel(logging.INFO); log.addHandler(handler)
    try:
        for epoch in range(start,cfg['epochs']):
            if bad >= cfg['patience']: break
            tr = epoch_pass(model,train,name,device,optimizer,cfg['grad_clip'])
            va = epoch_pass(model,validation,name,device)
            improved = va['mae'] < best-cfg['min_delta']
            best, bad = (va['mae'],0) if improved else (best,bad+1)
            if improved:
                atomic_save(dict(model=model.state_dict(),epoch=epoch,validation=va),out/'best.pth')
            state = dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,
                         best=best,bad=bad,python_rng=random.getstate(),numpy_rng=np.random.get_state(),
                         torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()
                         if device.type=='cuda' else [],loader_rng=generator.get_state())
            atomic_save(state,resume)
            text = f'{name} epoch={epoch+1} train={tr} validation={va} best={best:.6f} patience={bad}'
            log.info(text)
            # One JSON per committed epoch: resume never duplicates history rows.
            (out/f'epoch_{epoch+1:04d}.json').write_text(json.dumps(dict(train=tr,validation=va)),encoding='utf-8')
            history = [json.loads(p.read_text(encoding='utf-8')) | {'epoch':int(p.stem.split('_')[1])}
                       for p in sorted(out.glob('epoch_*.json'))]
            (out/'training_history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
            with (out/'training_history.csv').open('w',newline='',encoding='utf-8') as f:
                w = csv.DictWriter(f,fieldnames=['epoch','train_loss','train_mae','validation_loss','validation_mae'])
                w.writeheader()
                for row in history:
                    w.writerow(dict(epoch=row['epoch'],train_loss=row['train']['loss'],train_mae=row['train']['mae'],
                                    validation_loss=row['validation']['loss'],validation_mae=row['validation']['mae']))
        model.load_state_dict(load_checkpoint(out/'best.pth','cpu')['model'])
        test = loader(directory,'test',name,cfg)
        with (out/'test_predictions.csv').open('w',newline='',encoding='utf-8') as f:
            writer = csv.writer(f); writer.writerow(['key','truth','prediction'])
            metrics = epoch_pass(model,test,name,device,writer=writer)
        metrics.update(task=task_for(name),best_validation_mae=best,
                       parameters=sum(p.numel() for p in model.parameters()))
        (out/'completed.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    finally:
        log.removeHandler(handler); handler.close()


def main():
    global MEMORY_BANK
    p = argparse.ArgumentParser()
    p.add_argument('--check',action='store_true',help='synthetic four-model checks only, no dataset required')
    p.add_argument('--new-run',action='store_true')
    args = p.parse_args()
    cfg = json.loads((HERE/'train_config.json').read_text(encoding='utf-8'))
    cpu_threads = cfg.get('cpu_threads',4)
    if not isinstance(cpu_threads,int) or cpu_threads < 1:
        raise ValueError('cpu_threads必须为正整数')
    torch.set_num_threads(cpu_threads)
    torch.set_num_interop_threads(1)
    allowed = {'seist_s_emg','seist_s_dis','seist_s_baz','magnet'}
    if not cfg['models'] or len(set(cfg['models']))!=len(cfg['models']) or not set(cfg['models'])<=allowed:
        raise ValueError('无效模型列表')
    if any(cfg[k]<1 for k in ('batch_size','epochs','patience')) or cfg['workers']<0:
        raise ValueError('无效训练配置')
    if args.check:
        preflight(cfg); return
    if cfg['device'].startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA不可用；不自动切换CPU')
    directory = resolve_dataset(cfg['dataset_dir'])
    identity = dict(config=cfg,dataset=str(directory),dataset_manifest_sha256=sha(directory/'manifest.json'),
                    cohort_sha256=sha(directory/'cohort.npz'),
                    code_sha256={str(f.relative_to(HERE)):sha(f) for f in
                    [Path(__file__),HERE/'comparison_dataset.py',*UPSTREAM.glob('*.py')]})
    results = HERE/'training_results'; results.mkdir(exist_ok=True)
    active = results/'active_run.json'
    if active.exists() and not args.new_run:
        run = Path(json.loads(active.read_text())['directory'])
        previous = json.loads((run/'identity.json').read_text(encoding='utf-8'))
        if previous != identity:
            old_hashes = json.loads((HERE/'pre_memory_code_hashes.json').read_text(encoding='utf-8-sig'))
            expected = dict(identity)
            expected['code_sha256'] = dict(identity['code_sha256'])
            expected['code_sha256'].update(old_hashes)
            receipt_path = run/'memory_loading_compatibility.json'
            receipt = json.loads(receipt_path.read_text(encoding='utf-8')) if receipt_path.exists() else {}
            compatible = (receipt.get('previous_identity_sha256')==sha(run/'identity.json')
                          and receipt.get('runtime_identity')==identity)
            if previous != expected and not compatible:
                raise RuntimeError('配置、代码或数据改变；需--new-run，禁止混入旧结果')
    else:
        preflight(cfg,directory)
        run = results/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        run.mkdir()
        (run/'identity.json').write_text(json.dumps(identity,indent=2),encoding='utf-8')
        active.write_text(json.dumps(dict(directory=str(run))),encoding='utf-8')
    # Windows OS lock is released automatically if a process crashes.
    import msvcrt
    with (run/'run.lock').open('a+b') as lock:
        lock.seek(0); lock.write(b'0'); lock.flush(); lock.seek(0)
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        identity_path = run/'identity.json'
        if json.loads(identity_path.read_text(encoding='utf-8')) != identity:
            (run/'identity_before_memory_loading.json').write_text(
                identity_path.read_text(encoding='utf-8'),encoding='utf-8')
            identity_path.write_text(json.dumps(identity,indent=2),encoding='utf-8')
        logging.basicConfig(force=True,level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',
            handlers=[logging.StreamHandler(),logging.FileHandler(run/'experiment.log',encoding='utf-8')])
        MEMORY_BANK = load_memory_bank(directory,processes=cfg.get('h5_read_processes',16),
                                       audit_path=run/'split_audit.json')
        for name in cfg['models']:
            task_dir = run/name
            task_dir.mkdir(exist_ok=True)
            running = task_dir/'running.json'
            failed = task_dir/'failed.json'
            completed = task_dir/'completed.json'
            if completed.exists():
                print(f'跳过已完成任务 {name}',flush=True)
                continue
            if failed.exists():
                print(f'跳过此前报错任务 {name}',flush=True)
                continue
            running.write_text(json.dumps(dict(started=datetime.now().isoformat())),encoding='utf-8')
            try:
                train_model(name,cfg,directory,run)
            except Exception:
                details = traceback.format_exc()
                failed.write_text(json.dumps(dict(status='failed',error=details),
                                             ensure_ascii=False,indent=2),encoding='utf-8')
                print(f'{name} 失败，已记录错误，继续下一个任务。\n{details}',flush=True)
            finally:
                # KeyboardInterrupt/SystemExit propagate without a failure marker;
                # a later launch resumes the last fully committed epoch.
                running.unlink(missing_ok=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        summary = {}
        for name in cfg['models']:
            task_dir = run/name
            result = task_dir/'completed.json'
            if not result.exists():
                result = task_dir/'failed.json'
            summary[name] = json.loads(result.read_text(encoding='utf-8'))
        (run/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(f'任务调度结束（成功/失败详情见summary.json）：{run}',flush=True)


if __name__=='__main__':
    mp.freeze_support()
    main()
