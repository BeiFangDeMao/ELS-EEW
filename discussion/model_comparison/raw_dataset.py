"""Worker-local, read-only HDF5 access for the frozen comparison cohort."""
import json
from pathlib import Path

import h5py
import numpy as np
from torch.utils.data import Dataset


class RawComparisonDataset(Dataset):
    """Return unprocessed ZNE gal arrays; preprocessing belongs to each model.

    Labels are magnitude, epicentral distance (km), station-to-epicenter
    azimuth (degrees), NOT a ready-to-use SeisT/MagNet loss target.
    """

    def __init__(self, split, directory=None, samples=2976):
        if split not in ('train', 'validation', 'test'):
            raise ValueError('Unknown split')
        if not isinstance(samples, int) or not 1 <= samples <= 2976:
            raise ValueError('samples must be in [1, 2976]')
        base = Path(__file__).resolve().parent / 'data'
        directory = Path(directory or json.loads(
            (base/'active_dataset.json').read_text(encoding='utf-8'))['directory'])
        manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
        self.source = Path(manifest['raw_h5'])
        stat = self.source.stat()
        if (stat.st_size, stat.st_mtime_ns) != (
                manifest['raw_h5_size'], manifest['raw_h5_mtime_ns']):
            raise RuntimeError('Raw H5 changed since data preparation')
        with np.load(directory/'cohort.npz', allow_pickle=False) as cohort:
            selected = cohort['splits'] == split
            self.keys = cohort['keys'][selected].copy()
            self.labels = cohort['labels'][selected].copy()
        self.samples = samples
        self._file = None

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        if self._file is None:
            self._file = h5py.File(self.source, 'r')
        key = str(self.keys[index])
        wave = self._file[key][:self.samples].astype(np.float32).T.copy()
        if wave.shape != (3, self.samples) or not np.isfinite(wave).all():
            raise ValueError(f'Invalid waveform {key}')
        return {'waveform': wave, 'labels': self.labels[index].copy(), 'key': key}

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_file'] = None
        return state
