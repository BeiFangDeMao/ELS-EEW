"""CPU-only measurement of one true test record, D0 vs D1. Never trains."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import h5py
import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
ORIGINAL_RUN = HERE / 'results/20261021_003609_054421_seed42'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from tensors(item)


def state_bytes(state):
    values = list(tensors(state))
    logical = sum(t.numel() * t.element_size() for t in values)
    # Count backing allocations, not only view sizes, with shared storage deduplicated.
    storage = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes() for t in values if t.numel()}
    return logical, sum(storage.values())


def update(model, mode, chunk, state):
    if mode == 'D0':
        history = chunk.clone() if state is None else torch.cat((state, chunk), dim=-1)
        return model(history)[0][:, -1:, :], history
    return model.stream_step(chunk, state)


def replay(model, mode, wave, collect=False):
    state, delays, outputs = None, [], []
    cpu_start = time.process_time_ns()
    for chunk in wave.split(32, dim=-1):
        start = time.perf_counter_ns()
        prediction, state = update(model, mode, chunk, state)
        elapsed = (time.perf_counter_ns() - start) / 1e6
        delays.append(elapsed)
        if collect:
            outputs.append(prediction.clone())
    cpu_ms = (time.process_time_ns() - cpu_start) / 1e6
    return delays, cpu_ms, torch.cat(outputs, dim=1) if collect else None


def mac_hooks(model):
    counter = {'macs': 0}
    handles = []
    def conv(module, inputs, output):
        counter['macs'] += output.numel() * (module.in_channels // module.groups) * module.kernel_size[0]
    def gru(module, inputs, output):
        for layer in range(module.num_layers):
            ins = module.input_size if layer == 0 else module.hidden_size
            counter['macs'] += inputs[0].shape[0] * inputs[0].shape[1] * 3 * module.hidden_size * (ins + module.hidden_size)
    def linear(module, inputs, output):
        counter['macs'] += output.numel() * module.in_features
    for module in model.modules():
        hook = conv if isinstance(module, nn.Conv1d) else gru if isinstance(module, nn.GRU) else linear if isinstance(module, nn.Linear) else None
        if hook:
            handles.append(module.register_forward_hook(hook))
    return counter, handles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test-index', type=int, default=0)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--warmups', type=int, default=2)
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmups < 1 or args.threads < 1:
        raise ValueError('Positive repeats, warmups, threads required')
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    cfg_path = ORIGINAL_RUN / 'ELS8K_Full/config.json'
    cfg = json.loads(cfg_path.read_text(encoding='utf-8'))['experiment']
    weights = Path(cfg['reference_checkpoint'])
    weights_hash = digest(weights)
    assert weights_hash == cfg['reference_sha256']
    sys.path.insert(0, cfg['project'])
    from models.compact_gru import CompactGRU
    with np.load(cfg['label_cache'], allow_pickle=False) as cached:
        keys = cached['keys'][cached['splits'] == 'test']
        assert len(keys) == 51772 and 0 <= args.test_index < len(keys)
        key = str(keys[args.test_index])
    with h5py.File(cfg['dataset'], 'r') as file:
        ds = file[key]
        assert ds.attrs['sampling_freq_hz'] == 100
        assert ds.attrs['channel_order'] == 'Z,N,E' and ds.attrs['waveform_unit'] == 'gal'
        wave = torch.from_numpy(np.ascontiguousarray(ds[:].astype(np.float32).T)).unsqueeze(0)
    assert wave.shape == (1, 3, 2976) and torch.isfinite(wave).all()
    checkpoint = torch.load(weights, map_location='cpu', weights_only=False)
    model = CompactGRU().cpu().float().eval()
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    del checkpoint
    assert all(t.device.type == 'cpu' for t in model.parameters())
    assert sum(p.numel() for p in model.parameters()) == 8263
    folder = HERE / 'cpu_single_record_results' / (datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_CPU_D0_D1')
    folder.mkdir(parents=True)
    print(f'CPU only; threads={args.threads}; record={key}; 93 updates; FP32; batch=1', flush=True)
    update_rows, consistency = [], []
    delays = {mode: [] for mode in ('D0', 'D1')}
    cpu_times = {mode: [] for mode in ('D0', 'D1')}
    with torch.inference_mode():
        reference = model(wave)[0]
        for mode in ('D0', 'D1'):
            _, _, predictions = replay(model, mode, wave, collect=True)
            torch.testing.assert_close(reference, predictions, rtol=2e-4, atol=2e-5)
            consistency.append(dict(mode=mode, passed=True, max_absolute_difference=float((reference - predictions).abs().max()),
                                    max_difference_by_output=(reference - predictions).abs().amax(dim=(0, 1)).tolist()))
            # Resource pass is separated from all latency timers.
            counts, handles = mac_hooks(model)
            state = None
            try:
                for j, chunk in enumerate(wave.split(32, dim=-1)):
                    counts['macs'] = 0
                    _, state = update(model, mode, chunk, state)
                    logical, allocated = state_bytes(state)
                    assert logical == allocated, 'Unexpected backing-storage retention'
                    if mode == 'D0':
                        raw_bytes, conv_bytes, gru_bytes = allocated, 0, 0
                        assert allocated == 3 * 32 * (j + 1) * 4
                        assert counts['macs'] == 243296 * (j + 1)
                    else:
                        raw_bytes = 0
                        conv_bytes = state_bytes(state[0])[1]
                        gru_bytes = state_bytes(state[1])[1]
                        assert allocated == 2112 and counts['macs'] == 243296
                    update_rows.append(dict(strategy=mode, output_chunk=j + 1, observation_time_s=(j + 1) * .32,
                                            macs_per_update=counts['macs'], input_history_bytes=raw_bytes,
                                            convolution_cache_bytes=conv_bytes, gru_state_bytes=gru_bytes,
                                            persistent_cache_logical_bytes=logical, persistent_cache_storage_bytes=allocated))
            finally:
                for handle in handles:
                    handle.remove()
            print(f'{mode}: output consistency and actual cache allocations checked; warmup', flush=True)
            for _ in range(args.warmups):
                replay(model, mode, wave)
        # Alternate order across repetitions to reduce fixed-order timing bias.
        for repeat in range(args.repeats):
            for mode in (('D0', 'D1') if repeat % 2 == 0 else ('D1', 'D0')):
                measured, cpu_ms, _ = replay(model, mode, wave)
                delays[mode].append(measured)
                cpu_times[mode].append(cpu_ms)
                print(f'Repeat {repeat + 1}/{args.repeats} {mode}: {sum(measured):.3f} ms / record', flush=True)
    summary, latency_rows = [], []
    for mode in ('D0', 'D1'):
        samples = np.asarray(delays[mode])
        resources = [r for r in update_rows if r['strategy'] == mode]
        totals = samples.sum(axis=1)
        total_macs = sum(r['macs_per_update'] for r in resources)
        final = resources[-1]
        summary.append(dict(strategy=mode, test_record=key, distinct_test_records=1, timing_repeats=args.repeats,
                            cpu_threads=args.threads, batch_size=1, observation_window_s=29.76, updates=93,
                            median_update_ms=float(np.median(samples)), p95_update_ms=float(np.percentile(samples, 95)),
                            first_update_median_ms=float(np.median(samples[:, 0])), final_update_median_ms=float(np.median(samples[:, -1])),
                            median_record_total_ms=float(np.median(totals)), p95_record_total_ms=float(np.percentile(totals, 95)),
                            median_record_cpu_ms=float(np.median(cpu_times[mode])),
                            total_macs=total_macs, total_mmac=total_macs / 1e6,
                            first_update_macs=resources[0]['macs_per_update'], final_update_macs=final['macs_per_update'],
                            average_realtime_demand_mmac_per_s=total_macs / 1e6 / 29.76,
                            final_update_realtime_demand_mmac_per_s=final['macs_per_update'] / 1e6 / .32,
                            input_history_kib=final['input_history_bytes'] / 1024,
                            convolution_cache_kib=final['convolution_cache_bytes'] / 1024,
                            gru_state_kib=final['gru_state_bytes'] / 1024,
                            final_persistent_cache_kib=final['persistent_cache_storage_bytes'] / 1024,
                            peak_persistent_cache_kib=max(r['persistent_cache_storage_bytes'] for r in resources) / 1024,
                            current_input_chunk_kib=wave[:, :, :32].numel() * wave.element_size() / 1024,
                            fp32_model_weights_kib=sum(p.numel() * p.element_size() for p in model.parameters()) / 1024))
        for repeat, values in enumerate(samples):
            for j, elapsed in enumerate(values):
                latency_rows.append(dict(strategy=mode, repeat=repeat + 1, output_chunk=j + 1,
                                         observation_time_s=(j + 1) * .32, latency_ms=float(elapsed)))
        np.save(folder / f'{mode}_latency_samples_ms.npy', samples)
    assert digest(weights) == weights_hash
    result = dict(summary=summary, updates=update_rows, latency_samples=latency_rows,
                  environment=dict(cpu=platform.processor(), platform=platform.platform(), torch=torch.__version__,
                                   device='cpu', cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
                                   torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
                                   fp32=True, mkldnn_enabled=torch.backends.mkldnn.enabled,
                                   warmup_records_per_mode=args.warmups, timing_scope='model plus causal history concatenation; excludes disk, preprocessing, logging, resource hooks and output accumulation'),
                  consistency=consistency, checkpoint=str(weights), checkpoint_sha256=weights_hash,
                  source_sha256={str(cfg_path): digest(cfg_path), str(Path(__file__).resolve()): digest(Path(__file__))},
                  cache_scope='Actual retained tensor backing storage between calls. Excludes weights, current input chunk, temporary activations, runtime workspace, Python and OS memory; not total process RAM.',
                  macs_scope='Runtime Conv1d/GRU/Linear shape hooks, including all 32 GRU samples. Excludes nonlinearities, normalization, elementwise gates, bias additions and copies. Not measured CPU hardware instructions.')
    (folder / 'measurement.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    report = ['# 单条真实测试波形：CPU实测D0与D1', f'记录：`{key}`。FP32、batch=1、CPU {args.threads}线程；同一条记录重复{args.repeats}次，预热每种方式{args.warmups}次。不是多条记录的总体分布。',
              '只读取原8K最佳权重，无训练、微调或GPU运算。输入沿用原H5的100 Hz、ZNE、gal、2976点因果预处理，不重新处理波形。',
              'D0每次接收32个新点，将已到达数据拼接到历史中后重算全部前缀；D1调用原stream_step，只保留卷积缓存和GRU状态。两者逐时刻原始输出一致性检查通过。',
              '## 本次测量', '|方式|更新中位数(ms)|整条记录中位数(ms)|总MACs(MMAC)|最大持久缓存(KiB)|', '|---|---:|---:|---:|---:|']
    for row in summary:
        report.append(f"|{row['strategy']}|{row['median_update_ms']:.4f}|{row['median_record_total_ms']:.4f}|{row['total_mmac']:.6f}|{row['peak_persistent_cache_kib']:.4f}|")
    report += ['## 缓存及算力口径', '缓存通过运行中真实Tensor的untyped_storage().nbytes()逐次读取，去重共享storage，并检查逻辑大小与底层分配一致。D0需保留已到达的原始波形；D1保留四层卷积的2/4/8/16点特征历史及两层GRU的24维隐藏状态。',
               'D0输入历史从0.375 KiB增长至34.875 KiB；D1持久状态恒定2.0625 KiB，其中卷积1.875 KiB、GRU 0.1875 KiB。',
               '这不是CPU硬件缓存(L1/L2/L3)，也不是进程峰值RAM。另列当前输入块0.375 KiB及模型权重32.27734375 KiB；临时激活、计算库workspace、解释器和系统内存不在持久缓存内。',
               'MACs由真实运行的卷积、GRU、Linear算子形状计数；不是硬件指令计数。每个MAC按一次乘法加一次加法计约2 FLOPs，仅作为换算，不能称为包含全部操作的实测FLOPs。',
               '总MACs为93次更新之和；单次延迟为5次重复×93个时刻的汇总中位数。D0随时间增长，详见逐时刻资源与延迟CSV，不能把35.92 ms等既有多记录结果当作本次实测。',
               '时延只包含模型及必要的历史拼接，排除文件读取、离线预处理、日志、资源统计钩子和输出积累。现有权重和实验结果不覆盖。']
    (folder / 'summary.md').write_text('\n\n'.join(report), encoding='utf-8')
    (HERE / 'cpu_single_record_results/latest_run.json').write_text(json.dumps(dict(result_directory=str(folder), record_id=key), ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(result_directory=str(folder), summary=summary), ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
