"""Small synthetic check only; never reads the user's waveform files."""
import tempfile
from pathlib import Path

import h5py
import numpy as np

from build_dataset_288 import build, make_processor, process_window


def main():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        keys = np.array(['ABCDEF180101', 'ABCDEF200101', 'ABCDEF230101'])
        splits = np.array(['train', 'validation', 'test'])
        labels = np.array([[4., 10., 20.]] * 3)
        wave = np.arange(900, dtype=np.float32).reshape(300, 3)
        expected_wave = process_window(wave[:288], make_processor())
        streaming = make_processor()
        blocks = [streaming.step(wave[i:i+32].T[None, :, :])[0].T
                  for i in range(0, 288, 32)]
        np.testing.assert_array_equal(expected_wave, np.concatenate(blocks))
        with h5py.File(root/'raw.h5', 'w') as f:
            f.attrs.update(channel_order='Z,N,E', unit='gal', sampling_freq_hz=100,
                           filtered=False, normalized=False, detrended=False)
            for key, label in zip(keys, labels):
                ds = f.create_dataset(key, data=wave)
                ds.attrs.update(mag=label[0], dis=label[1], azi=label[2],
                                sampling_freq_hz=100, sample_count=300)
        np.savez(root/'cohort.npz', keys=keys, labels=labels, splits=splits)
        for processes in (1, 2):
            out = build(root/'raw.h5', root/'cohort.npz', root, processes,
                        dict(train=1, validation=1, test=1))
            with h5py.File(out/'SeisT_MagNet_288.h5', 'r') as f:
                for key, split in zip(keys, splits):
                    np.testing.assert_array_equal(f[key][:], expected_wave)
                    assert f[key].attrs['split'] == split
                    assert f[key].attrs['sample_count'] == 288
                    assert f[key].attrs['mag'] == 4.
                assert f.attrs['filtered']
                assert f.attrs['processing_call_samples'] == 288
        print('PASS: synthetic serial/spawn, full-window/streaming equivalence, labels and metadata')


if __name__ == '__main__':
    main()
