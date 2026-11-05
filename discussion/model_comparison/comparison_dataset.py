"""Read the completed 288-point causal H5 without further preprocessing."""
import json
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

HERE = Path(__file__).resolve().parent


def load_memory_bank(directory, processes=16, audit_path=None):
    """Same ordered parallel H5 reader used by CompactGRU/ablation."""
    import sys
    import hashlib
    import logging
    sys.path.insert(0, str(HERE.parents[1]/'els_eew/training'))
    from h5_loading import load_h5_records
    canonical = HERE.parents[1]/'plotting/Fig01_Data_Distribution/labels.npz'
    with np.load(canonical, allow_pickle=False) as c:
        keys, labels, saved_splits = (c[n].copy() for n in ('keys','labels','splits'))
    with np.load(Path(directory)/'cohort.npz', allow_pickle=False) as c:
        for name, expected in [('keys',keys),('labels',labels),('splits',saved_splits)]:
            np.testing.assert_array_equal(c[name], expected)
    records, stats = load_h5_records(
        Path(directory)/'SeisT_MagNet_288.h5', sequence_length=288,
        read_processes=processes, exclude_200hz=True, allow_padding=False)
    if stats['error_count']:
        raise RuntimeError(f'H5读取错误：{stats}')
    np.testing.assert_array_equal([r[2] for r in records], keys)
    splits = np.array(['validation' if k[6:8] in ('19','20','21','22')
                       else 'test' if k[6:8] in ('23','24') else 'train' for k in keys])
    np.testing.assert_array_equal(splits, saved_splits)
    waves = np.empty((len(keys),3,288),dtype=np.float32)
    for i,(wave, target, key) in enumerate(records):
        angle = labels[i,2]*np.pi/180
        expected = np.array([labels[i,0],labels[i,1],np.sin(angle),np.cos(angle)],dtype=np.float32)
        np.testing.assert_allclose(target,expected,rtol=0,atol=1e-6)
        waves[i] = wave
        records[i] = None
    del records
    expected_counts = dict(train=605166,validation=84564,test=51772)
    audit = {}
    for name, count in expected_counts.items():
        selected = keys[splits==name]
        if len(selected)!=count:
            raise RuntimeError(f'{name}记录数量错误：{len(selected)}')
        audit[name] = dict(records=count, ordered_keys_sha256=hashlib.sha256(
            '\n'.join(selected).encode()).hexdigest())
    audit['loader_stats'] = stats
    if audit_path:
        Path(audit_path).write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    logging.info('全部读取进程已退出；波形占用 %.2f GiB；CompactGRU原划分核对通过：%s',
                 waves.nbytes/1024**3,expected_counts)
    return dict(waves=waves,keys=keys,labels=labels.astype(np.float32),splits=splits)


class MemoryComparisonDataset(Dataset):
    def __init__(self, bank, split, task):
        self.bank = bank
        self.indices = np.flatnonzero(bank['splits']==split)
        self.column = {'emg':0,'dis':1,'baz':2}[task]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self,index):
        i = self.indices[index]
        return (self.bank['waves'][i],self.bank['labels'][i,self.column:self.column+1],
                str(self.bank['keys'][i]))


def resolve_dataset(value='auto'):
    if value != 'auto':
        path = Path(value)
        if not path.is_absolute():
            path = HERE / path
        path = path.resolve()
    else:
        candidates = []
        for p in (HERE/'data').glob('SeisT_MagNet_288_Causal_*/manifest.json'):
            m = json.loads(p.read_text(encoding='utf-8-sig'))
            if m.get('status') == 'completed':
                candidates.append(p.parent)
        if not candidates:
            raise FileNotFoundError('没有完成的288点因果数据集，请等待制作完成或设置dataset_dir')
        path = sorted(candidates)[-1]
    manifest = json.loads((path/'manifest.json').read_text(encoding='utf-8-sig'))
    if (manifest.get('status') != 'completed' or manifest.get('samples') != 288
            or manifest.get('sampling_rate_hz') != 100):
        raise ValueError('需要完整制作完成的100Hz/288点数据集')
    h5 = path/'SeisT_MagNet_288.h5'
    with h5py.File(h5, 'r') as f:
        if (not f.attrs.get('causal_processed', False) or f.attrs.get('channel_order') != 'Z,N,E'
                or f.attrs.get('unit') != 'gal'):
            raise ValueError('训练要求288点因果预处理H5，不能使用早期Raw裁剪版')
    return path


class ComparisonDataset(Dataset):
    def __init__(self, directory, split, task):
        if split not in ('train', 'validation', 'test') or task not in ('emg','dis','baz'):
            raise ValueError('无效划分/任务')
        self.path = str(Path(directory)/'SeisT_MagNet_288.h5')
        with np.load(Path(directory)/'cohort.npz', allow_pickle=False) as c:
            mask = c['splits'] == split
            self.keys = c['keys'][mask].copy()
            self.targets = c['labels'][mask, {'emg':0,'dis':1,'baz':2}[task]].astype(np.float32)
        if len(self.keys) == 0 or not np.isfinite(self.targets).all():
            raise ValueError('空数据或无效标签')
        self.split = split
        self.label_attribute = {'emg':'mag','dis':'dis','baz':'azi'}[task]
        self._file = None

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        if self._file is None:
            self._file = h5py.File(self.path, 'r')
        key = str(self.keys[index])
        record = self._file[key]
        if record.attrs['split'] != self.split:
            raise ValueError(f'H5与清单划分不一致：{key}')
        if not np.isclose(float(record.attrs[self.label_attribute]), float(self.targets[index]),
                          rtol=1e-6, atol=1e-5):
            raise ValueError(f'H5与清单标签不一致：{key}')
        wave = record[:].astype(np.float32).T.copy()
        if wave.shape != (3,288) or not np.isfinite(wave).all():
            raise ValueError(f'无效波形：{key}')
        return wave, self.targets[index:index+1].copy(), key

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_file'] = None
        return state

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None
