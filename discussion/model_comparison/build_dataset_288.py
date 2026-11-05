# -*- coding: utf-8 -*-
"""IDE直接运行：Raw H5 -> SeisT/MagNet 共用288点H5。不会启动训练。

参考 dataset/制作所有日本强震数据因果.py 的 Windows spawn、独立只读句柄、
有界批次、主进程独占写入和 partial 文件流程。
复用CompactGRU预处理：首点基线扣除+0.75–40Hz单向SOS，一次处理288点。
"""
import atexit
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

# ---------- 修改配置后直接运行 ----------
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INPUT_H5 = ROOT / 'dataset/dataseth5/J_K_KIKnet_3000_Raw_100Hz.h5'
COHORT_FILE = ROOT / 'plotting/Fig01_Data_Distribution/labels.npz'
OUTPUT_PARENT = HERE / 'data'
NUM_PROCESSES = 16
BATCH_SIZE = 256
MAX_IN_FLIGHT = 32
TARGET_LENGTH = 288
SAMPLING_RATE = 100
COMPRESSION = None  # None 或 'lzf'
PREPROCESSOR_FILE = ROOT / 'els_eew/training/causal_preprocessing.py'

_SOURCE = None
_PROCESSOR = None


def make_processor():
    spec = importlib.util.spec_from_file_location('_comparison_causal', PREPROCESSOR_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CausalWavePreprocessor(sampling_rate=100., band=(.75, 40.), order=4,
                                         mode='causal_bandpass', subtract_initial=True)


def process_window(wave, processor):
    """(288,3) -> (288,3); independent zero state and a single step call.

    Same SOS recurrence as 9 consecutive 32-point blocks with carried state.
    Reads no samples after index287. No full-window mean or backward filtering.
    """
    if wave.shape != (TARGET_LENGTH, 3):
        raise ValueError('要求(288,3)窗口')
    processor.reset()
    return processor.step(wave.T[None, :, :])[0].T.copy()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def close_worker():
    global _SOURCE
    if _SOURCE is not None:
        _SOURCE.close()
        _SOURCE = None


def initialize_worker(source):
    global _SOURCE, _PROCESSOR
    _SOURCE = h5py.File(source, 'r')
    _PROCESSOR = make_processor()
    atexit.register(close_worker)


def read_batch(task):
    result = []
    for key, label, split in task:
        try:
            ds = _SOURCE[key]
            if ds.ndim != 2 or ds.shape[1] != 3 or ds.shape[0] < TARGET_LENGTH:
                raise ValueError('要求(N,3)，且N>=288；不补零')
            if float(ds.attrs['sampling_freq_hz']) != SAMPLING_RATE:
                raise ValueError('当前采样率必须为100Hz')
            wave = ds[:TARGET_LENGTH].astype(np.float32)
            if not np.isfinite(wave).all():
                raise ValueError('波形含非有限值')
            wave = process_window(wave, _PROCESSOR)
            np.testing.assert_allclose([ds.attrs[n] for n in ('mag', 'dis', 'azi')],
                                       label, rtol=0, atol=1e-10)
            attrs = []
            for name in ds.attrs:
                dtype = ds.attrs.get_id(name).dtype
                if h5py.check_dtype(ref=dtype) is not None:
                    raise ValueError(f'不支持跨文件引用属性{name}')
                attrs.append((name, ds.attrs[name], dtype))
            result.append((key, wave, attrs, split))
        except Exception as exc:
            raise RuntimeError(f'记录{key}处理失败：{exc}') from exc
    return result


def batches(source, tasks, processes):
    if processes == 1:
        initialize_worker(source)
        try:
            for task in tasks:
                yield read_batch(task)
        finally:
            close_worker()
        return
    with ProcessPoolExecutor(max_workers=processes, mp_context=mp.get_context('spawn'),
                             initializer=initialize_worker, initargs=(source,)) as pool:
        pending = set()
        try:
            for _ in range(MAX_IN_FLIGHT):
                task = next(tasks, None)
                if task is None:
                    break
                pending.add(pool.submit(read_batch, task))
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    yield future.result()
                    task = next(tasks, None)
                    if task is not None:
                        pending.add(pool.submit(read_batch, task))
        finally:
            for future in pending:
                future.cancel()


def build(input_h5=INPUT_H5, cohort_file=COHORT_FILE, output_parent=OUTPUT_PARENT,
          processes=NUM_PROCESSES, expected_counts=None):
    if expected_counts is None:
        expected_counts = {'train': 605166, 'validation': 84564, 'test': 51772}
    if processes < 1 or BATCH_SIZE < 1 or MAX_IN_FLIGHT < 1:
        raise ValueError('进程数、批次和队列上限必须为正整数')
    if COMPRESSION not in (None, 'lzf'):
        raise ValueError('压缩仅支持None/lzf')
    input_h5, cohort_file = Path(input_h5).resolve(), Path(cohort_file).resolve()
    with np.load(cohort_file, allow_pickle=False) as c:
        keys, labels, splits = (c[n].copy() for n in ('keys', 'labels', 'splits'))
    if (labels.shape != (len(keys), 3) or splits.shape != keys.shape
            or len(set(keys)) != len(keys) or not np.isfinite(labels).all()):
        raise ValueError('样本清单格式错误/重复键/无效标签')
    counts = {s: int(np.sum(splits == s)) for s in expected_counts}
    if counts != expected_counts or sum(counts.values()) != len(keys):
        raise ValueError(f'划分数量不符：{counts}')
    for k, s in zip(keys, splits):
        expected = ('validation' if k[6:8] in ('19','20','21','22') else
                    'test' if k[6:8] in ('23','24') else 'train')
        if s != expected:
            raise ValueError(f'年份划分不符：{k}')
    with h5py.File(input_h5, 'r') as src:
        if (src.attrs.get('channel_order') != 'Z,N,E' or src.attrs.get('unit') != 'gal'
                or float(src.attrs.get('sampling_freq_hz', 0)) != SAMPLING_RATE):
            raise ValueError('源H5必须是100Hz/Z,N,E/gal')
        for name in ('filtered', 'normalized', 'detrended'):
            if name not in src.attrs or bool(src.attrs[name]):
                raise ValueError(f'必须使用原始未处理H5：{name}')
    source_stat = input_h5.stat()
    out = Path(output_parent)/('SeisT_MagNet_288_Causal_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True, exist_ok=False)
    partial = out/'SeisT_MagNet_288.partial.h5'
    final = out/'SeisT_MagNet_288.h5'
    manifest_path = out/'manifest.json'
    manifest = dict(status='running', source=str(input_h5), source_size=source_stat.st_size,
                    source_mtime_ns=source_stat.st_mtime_ns, cohort=str(cohort_file),
                    cohort_sha256=digest(cohort_file), builder_sha256=digest(__file__),
                    counts=counts, samples=288, sampling_rate_hz=100, duration_s=2.88,
                    shape=[288,3], channel_order=['Z','N','E'], unit='gal',
                    processing='first-sample subtraction + causal SOS; one 288-point call per record',
                    preprocessing=make_processor().describe(),
                    preprocessor_file=str(PREPROCESSOR_FILE),
                    preprocessor_sha256=digest(PREPROCESSOR_FILE),
                    models=['SeisT','MagNet'], aggregation='waveform record',
                    event_disjointness_verified=False, written=0)
    def save_manifest():
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    save_manifest()
    tasks = (list(zip(keys[i:i+BATCH_SIZE].tolist(), labels[i:i+BATCH_SIZE],
                      splits[i:i+BATCH_SIZE].tolist())) for i in range(0, len(keys), BATCH_SIZE))
    try:
        with h5py.File(partial, 'x') as dst, tqdm(total=len(keys), desc='制作288点数据') as bar:
            dst.attrs.update(sampling_freq_hz=100, channel_order='Z,N,E', unit='gal',
                             duration_seconds=2.88, filtered=True, detrended=False,
                             normalized=False, causal_processed=True,
                             causal_band_hz=[.75, 40.], causal_butter_design_order=4,
                             causal_subtract_first_sample=True, processing_call_samples=288,
                             waveform_processing='first-sample subtraction; causal SOS; one 288-point call',
                             source_h5=str(input_h5), models='SeisT,MagNet')
            for batch in batches(str(input_h5), tasks, processes):
                for key, wave, attrs, split in batch:
                    ds = dst.create_dataset(key, data=wave, compression=COMPRESSION)
                    for name, value, dtype in attrs:
                        ds.attrs.create(name, value, dtype=dtype)
                    ds.attrs['sample_count'] = 288
                    ds.attrs['zne_array_shape'] = '3,288'
                    ds.attrs['window_duration_s'] = 2.88
                    ds.attrs['split'] = split
                    ds.attrs['preprocessing'] = 'first-sample subtraction; 0.75-40Hz causal SOS; one 288-point call'
                    ds.attrs['comparison_processing'] = 'CompactGRU preprocessing; full 288-point window'
                manifest['written'] += len(batch)
                bar.update(len(batch))
                dst.flush()
            if len(dst) != len(keys):
                raise RuntimeError('输出数量不符')
        if (input_h5.stat().st_size, input_h5.stat().st_mtime_ns) != (source_stat.st_size, source_stat.st_mtime_ns):
            raise RuntimeError('制作期间源H5发生变化')
        np.savez_compressed(out/'cohort.npz', keys=keys, labels=labels, splits=splits)
        partial.rename(final)
        manifest.update(status='completed', output=str(final), output_sha256=digest(final))
        save_manifest()
        print(f'完成：{final}', flush=True)
        return out
    except BaseException as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        save_manifest()
        raise


if __name__ == '__main__':
    mp.freeze_support()
    for var in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[var] = '1'
    build()
