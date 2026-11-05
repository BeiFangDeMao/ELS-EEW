# -*- coding: utf-8 -*-
"""Ordered parallel eager H5 loading; reader workers never import torch.

The parent lists/sorts keys once, closes that handle, and splits contiguous
balanced partitions. Each spawned reader opens its own read-only handle.
All readers are joined before results are assembled in original key order.
"""
import logging
import math
import multiprocessing as mp
import os
import time
from contextlib import contextmanager
from pathlib import Path
from queue import Empty

import h5py
import numpy as np
from tqdm import tqdm


def partition_keys(keys, processes):
    """Nonempty contiguous partitions, with sizes differing by at most one."""
    if not isinstance(processes, int) or isinstance(processes, bool) or processes < 1:
        raise ValueError("h5_read_processes必须为正整数；1表示串行读取")
    if not keys:
        return []
    count = min(processes, len(keys))
    size, extra = divmod(len(keys), count)
    partitions = []
    start = 0
    for i in range(count):
        stop = start + size + (i < extra)
        partitions.append(keys[start:stop])
        start = stop
    return partitions


@contextmanager
def _reader_environment():
    # Set before spawn imports NumPy, then restore the parent's environment.
    # Avoid 16 processes each starting a full set of BLAS/OpenMP threads.
    names = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    saved = {name: os.environ.get(name) for name in names}
    try:
        for name in names:
            os.environ[name] = "1"
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _read_partition(task, output_queue=None):
    """Top-level, picklable worker; no inherited HDF5 handle or CUDA state."""
    index, path, keys, length, exclude_200hz, allow_padding = task
    records = []
    # About 8 MiB per packet; do not serialize an entire GB-sized partition.
    packet_rows = max(1, min(256, (8 * 1024 * 1024) // (3 * length * 4 + 128)))
    packet_start = 0
    stats = dict(total=len(keys), valid=0, excluded_200hz=0, unknown_rate=0,
                 filtered_other=0, filtered_short=0, errors=[])
    # Opening the file itself failing aborts loading; it is not a bad record.
    with h5py.File(path, "r") as file:
        for position, key in enumerate(keys):
            try:
                group = file[key]
                if exclude_200hz:
                    try:
                        rate = float(group.attrs.get("sampling_freq_hz", float("nan")))
                    except (ValueError, TypeError):
                        rate = float("nan")
                    if math.isclose(rate, 200.0, rel_tol=0.0, abs_tol=1e-6):
                        stats['excluded_200hz'] += 1
                        continue
                    if not math.isfinite(rate):
                        stats['unknown_rate'] += 1
                # Check the entire stored record, matching the original loader.
                # Do not read only [:length]: that could miss invalid tail data.
                wave = group[:].astype(np.float32).T
                if wave.ndim != 2 or wave.shape[0] != 3:
                    stats['filtered_other'] += 1
                    continue
                mag = float(group.attrs.get("mag", np.nan))
                dis = float(group.attrs.get("dis", np.nan))
                azi = float(group.attrs.get("azi", np.nan))
                if (not np.isfinite([mag, dis, azi]).all() or not np.isfinite(wave).all()
                        or mag < 3 or dis >= 200):
                    stats['filtered_other'] += 1
                    continue
                if wave.shape[1] < length:
                    if not allow_padding:
                        stats['filtered_short'] += 1
                        continue
                    wave = np.pad(wave, ((0, 0), (0, length - wave.shape[1])), mode='constant')
                elif wave.shape[1] > length:
                    wave = wave[:, :length]
                # Copy only the retained portion into a compact contiguous array.
                wave = np.ascontiguousarray(wave, dtype=np.float32)
                angle = azi * math.pi / 180.0
                label = np.asarray([mag, dis, math.sin(angle), math.cos(angle)], dtype=np.float32)
                records.append((wave, label, key))
                stats['valid'] += 1
            except Exception as exc:
                stats['errors'].append((key, str(exc)))
            finally:
                if output_queue is not None and (
                        (position + 1) % packet_rows == 0 or position + 1 == len(keys)):
                    output_queue.put(('batch', index, packet_start,
                                      position + 1 - packet_start, records))
                    records = []
                    packet_start = position + 1
    if output_queue is not None:
        output_queue.put(('done', index, stats))
        return None
    return index, records, stats


def _read_to_queue(task, output_queue):
    try:
        _read_partition(task, output_queue)
    except BaseException as exc:
        output_queue.put(('fatal', task[0], f'{type(exc).__name__}: {exc}'))


def _parallel_read(tasks, bar):
    """Bounded small-packet transport; join every reader before returning."""
    count = len(tasks)
    context = mp.get_context('spawn')
    output_queue = context.Queue(maxsize=2 * count)
    processes = []
    records = [[] for _ in tasks]
    consumed = [0] * count
    stats = [None] * count
    try:
        with _reader_environment():
            for task in tasks:
                process = context.Process(target=_read_to_queue, args=(task, output_queue),
                                          name=f'H5Reader-{task[0] + 1}')
                process.start()
                processes.append(process)
            completed = 0
            while completed < count:
                try:
                    message = output_queue.get(timeout=1.0)
                except Empty:
                    failures = [(p.name, p.exitcode) for p in processes if p.exitcode not in (None, 0)]
                    if failures:
                        raise RuntimeError(f'H5读取进程异常退出：{failures}')
                    if all(not p.is_alive() for p in processes):
                        raise RuntimeError('H5读取进程全部退出，但数据未完整返回')
                    continue
                kind, index, *payload = message
                if kind == 'fatal':
                    raise RuntimeError(f'H5分片{index + 1}读取失败：{payload[0]}')
                if kind == 'batch':
                    start, processed, batch = payload
                    if start != consumed[index] or stats[index] is not None:
                        raise RuntimeError('H5回传批次顺序或数量错误')
                    records[index].extend(batch)
                    consumed[index] += processed
                    bar.update(processed)
                elif kind == 'done':
                    part_stats = payload[0]
                    if (stats[index] is not None or consumed[index] != part_stats['total']
                            or len(records[index]) != part_stats['valid']):
                        raise RuntimeError('H5分片回传数据不完整')
                    stats[index] = part_stats
                    completed += 1
                    bar.set_postfix(completed=completed, refresh=False)
                else:
                    raise RuntimeError(f'未知H5读取消息：{kind}')
            for process in processes:
                process.join()
                if process.exitcode != 0:
                    raise RuntimeError(f'{process.name}异常退出：{process.exitcode}')
    finally:
        # Covers Ctrl+C, failed file opens and unexpected worker termination.
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join()
            process.close()
        output_queue.close()
        output_queue.join_thread()
    return [(i, records[i], stats[i]) for i in range(count)]


def load_h5_records(h5_file_path, sequence_length, read_processes=16,
                    max_samples=None, exclude_200hz=False, show_progress=True, years=None,
                    allow_padding=True):
    """Return [(float32 wave(3,L), float32 label(4), key)], plus statistics.

    Defaults match the old SeismicDataset. Set allow_padding=False for replay:
    short records are excluded rather than inventing missing received samples.
    Returned NumPy records are converted to tensors only by the parent.
    """
    if not isinstance(sequence_length, int) or sequence_length < 1:
        raise ValueError("sequence_length必须为正整数")
    selected_years = None
    if years is not None:
        if isinstance(years, (str, bytes)):
            raise ValueError("years须为年份序列，如('23', '24')")
        year_items = tuple(map(str, years))
        if not year_items or not all(len(year) in (2, 4) and year.isdigit() for year in year_items):
            raise ValueError("years须为有效且不重复的两位或四位年份")
        selected_years = tuple(sorted({year[-2:] for year in year_items}))
        if len(selected_years) != len(year_items):
            raise ValueError("years须为有效且不重复的两位或四位年份")
    partition_keys([], read_processes)  # Validate even for an empty H5.
    started = time.perf_counter()
    path = str(Path(h5_file_path).resolve())
    logging.info("主进程枚举H5数据名称：%s", path)
    with h5py.File(path, "r") as file:
        keys = list(file.keys())
    # Preserve the old order of operations: limit first, then sort.
    if max_samples:
        keys = keys[:max_samples]
    keys.sort()
    listed_records = len(keys)
    if selected_years is not None:
        keys = [key for key in keys if (key[6:8] if len(key) >= 8 else '00') in selected_years]
        logging.info("按原文件名年份规则筛选%s年：%d/%d条，只读取选中年份的波形",
                     ','.join(selected_years), len(keys), listed_records)
    partitions = partition_keys(keys, read_processes)
    count = len(partitions)
    stats = dict(total=len(keys), valid=0, excluded_200hz=0, unknown_rate=0,
                 filtered_other=0, filtered_short=0, error_count=0, read_processes=count,
                 requested_processes=read_processes, elapsed_seconds=0.0)
    stats.update(listed_records=listed_records, excluded_year=listed_records - len(keys),
                 selected_years=selected_years)
    if not partitions:
        stats['elapsed_seconds'] = time.perf_counter() - started
        return [], stats
    logging.info("H5名称列表已就绪：%d条，分成%d份，每份%d–%d条", len(keys), count,
                 min(map(len, partitions)), max(map(len, partitions)))
    tasks = [(i, path, part, sequence_length, exclude_200hz, allow_padding)
             for i, part in enumerate(partitions)]
    parts = [None] * count
    with tqdm(total=len(keys), desc=f"读取H5（{count}份）", disable=not show_progress) as bar:
        if count == 1:
            result = _read_partition(tasks[0])
            parts[0] = result
            bar.update(result[2]['total'])
        else:
            # Windows-compatible spawn, with an import-safe standalone worker.
            # No training DataLoader workers are created by this pool.
            parts = _parallel_read(tasks, bar)
            logging.info("%d个H5读取进程已退出；主进程开始按原列表顺序合并", count)
    records = []
    # Completion order is irrelevant: contiguous partitions retain key order.
    for i in range(count):
        index, part_records, part_stats = parts[i]
        if index != i:
            raise RuntimeError("H5分片结果索引错误")
        records.extend(part_records)
        for name in ('valid', 'excluded_200hz', 'unknown_rate', 'filtered_other', 'filtered_short'):
            stats[name] += part_stats[name]
        stats['error_count'] += len(part_stats['errors'])
        for key, message in part_stats['errors']:
            logging.warning("跳过 %s: %s", key, message)
        parts[i] = None
    stats['elapsed_seconds'] = time.perf_counter() - started
    logging.info("H5读取合并完成：有效%d/%d条，总耗时%.2f秒", stats['valid'], stats['total'],
                 stats['elapsed_seconds'])
    return records, stats
