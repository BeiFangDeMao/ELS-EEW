"""Freeze the original CompactGRU cohort against raw waveform metadata.

Does not transform waveforms, train models or invent E3WS pre-P samples.
"""
import hashlib
import json
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RAW = ROOT / 'dataset/dataseth5/J_K_KIKnet_3000_Raw_100Hz.h5'
CACHE = ROOT / 'plotting/Fig01_Data_Distribution/labels.npz'
PROVENANCE = ROOT / 'plotting/Fig01_Data_Distribution/data_provenance.json'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    source_provenance = json.loads(PROVENANCE.read_text(encoding='utf-8'))
    canonical = Path(source_provenance['source_h5'])
    stat = canonical.stat()
    if (stat.st_size != source_provenance['source_size']
            or stat.st_mtime_ns != source_provenance['source_mtime_ns']):
        raise RuntimeError('Canonical source changed since cohort selection')
    with np.load(CACHE, allow_pickle=False) as cache:
        keys, labels, splits = (cache[k].copy() for k in ('keys', 'labels', 'splits'))
    if len(set(keys)) != len(keys) or not np.isfinite(labels).all():
        raise ValueError('Invalid cohort keys or labels')
    expected_splits = np.array([
        'validation' if k[6:8] in ('19', '20', '21', '22')
        else 'test' if k[6:8] in ('23', '24') else 'train' for k in keys
    ])
    np.testing.assert_array_equal(splits, expected_splits)
    counts = {s: int(np.sum(splits == s)) for s in ('train', 'validation', 'test')}
    if counts != source_provenance['counts']:
        raise ValueError(f'Cohort counts changed: {counts}')
    columns = ['file_prefix', 'mag', 'dis', 'azi', 'sampling_freq_hz',
               'sample_count', 'channel_order', 'waveform_unit', 'p_alignment_offset_s']
    metadata = pd.read_csv(RAW.with_suffix('.csv'), usecols=columns)
    if metadata.file_prefix.duplicated().any():
        raise ValueError('Duplicate raw metadata keys')
    metadata = metadata.set_index('file_prefix').loc[keys]
    np.testing.assert_allclose(metadata[['mag', 'dis', 'azi']].to_numpy(), labels,
                               rtol=0, atol=1e-10)
    if not ((metadata.sampling_freq_hz == 100).all()
            and (metadata.sample_count >= 2976).all()
            and (metadata.channel_order == 'Z,N,E').all()
            and (metadata.waveform_unit == 'gal').all()):
        raise ValueError('Unexpected raw waveform metadata')
    # Distributed sample checks: metadata/shape plus all values in sampled waves.
    sampled = []
    with h5py.File(RAW, 'r') as source:
        for split in counts:
            indices = np.flatnonzero(splits == split)
            for i in indices[np.linspace(0, len(indices)-1, 4, dtype=int)]:
                key = str(keys[i])
                record = source[key]
                wave = record[:]
                if wave.shape != (3000, 3) or not np.isfinite(wave).all():
                    raise ValueError(f'Invalid raw waveform: {key}')
                np.testing.assert_allclose([record.attrs[x] for x in ('mag','dis','azi')],
                                           labels[i], rtol=0, atol=1e-10)
                sampled.append(key)
    output = HERE / 'data' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / 'cohort.npz', keys=keys, labels=labels, splits=splits,
                        p_alignment_offset_s=metadata.p_alignment_offset_s.to_numpy())
    manifest = {
        'status': 'cohort_ready_model_preprocessing_pending',
        'raw_h5': str(RAW), 'raw_h5_size': RAW.stat().st_size,
        'raw_h5_mtime_ns': RAW.stat().st_mtime_ns,
        'canonical_causal_h5': str(canonical), 'counts': counts,
        'label_columns': ['mag', 'distance_km', 'station_to_epicenter_azimuth_deg'],
        'waveform': {'stored_shape': [3000, 3], 'common_post_p_samples': 2976,
                     'sampling_rate_hz': 100, 'channel_order': ['Z','N','E'], 'unit': 'gal'},
        'selection': 'Exact ordered CompactGRU cohort; no new filtering or random split',
        'aggregation': 'waveform records; event disjointness not established',
        'checks': {'all_selected_csv_labels_verified': True,
                   'raw_h5_sample_count': len(sampled), 'sampled_keys': sampled,
                   'all_raw_waveforms_scanned': False},
        'e3ws': 'pending separately rebuilt raw data with 7 s pre-P; do not pad missing pre-P',
        'preprocessing': 'Deferred per model; raw H5 unchanged, no duplicate waveforms written',
        'sha256': {str(p.relative_to(ROOT)): sha256(p) for p in
                   (CACHE, PROVENANCE, RAW.with_suffix('.csv'), Path(__file__), output/'cohort.npz')},
    }
    (output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                       encoding='utf-8')
    (HERE/'data/active_dataset.json').write_text(
        json.dumps({'directory': str(output)}, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(output), 'counts': counts,
                      'status': manifest['status']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
