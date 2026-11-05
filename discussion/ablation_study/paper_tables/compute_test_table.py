"""Derive the paper table from immutable, saved test predictions. No inference."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

HERE = Path(__file__).resolve().parent
RUN = HERE.parent / 'results' / '20261021_003609_054421_seed42'
TIMES = (2.88, 29.76)
HEADERS = ['Task', 'Model', '2.88 s MAE', '2.88 s R²', '29.76 s MAE', '29.76 s R²']
GROUPS = [
    ('Magnitude', 'magnitude', [('ELS-EEW', 'ELS8K_Full'), ('PureGRU', 'PureGRU_Existing'), ('Magnitude-only', 'Single_magnitude')]),
    ('Distance', 'distance', [('ELS-EEW', 'ELS8K_Full'), ('PureGRU', 'PureGRU_Existing'), ('Distance-only', 'Single_distance')]),
    ('Back-azimuth', 'azimuth', [('ELS-EEW', 'ELS8K_Full'), ('PureGRU', 'PureGRU_Existing'), ('Azimuth-only', 'Single_azimuth')]),
]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    baseline = np.load(RUN / 'ELS8K_Full' / 'truth_labels.npy')
    assert baseline.shape == (51772, 4)
    data, sources, checks = {}, {}, []
    for _, _, models in GROUPS:
        for _, name in models:
            if name in data:
                continue
            folder = RUN / name
            state = json.loads((folder / 'state.json').read_text(encoding='utf-8'))
            assert state['status'] == 'complete'
            pred = np.load(folder / 'predictions.npy', mmap_mode='r')
            truth = np.load(folder / 'truth_labels.npy')
            assert np.array_equal(truth, baseline), f'Label/order mismatch: {name}'
            assert pred.shape[:2] == (51772, 93)
            expected_dims = 4 if name in ('ELS8K_Full', 'PureGRU_Existing') else 2 if name == 'Single_azimuth' else 1
            assert pred.shape[2] == expected_dims
            metrics = pd.read_csv(folder / 'metrics_by_time.csv')
            assert len(metrics) == 93 and set(metrics.records) == {51772}
            data[name] = pred, truth.astype(np.float64), metrics
            for filename in ('predictions.npy', 'truth_labels.npy', 'metrics_by_time.csv', 'config.json', 'state.json'):
                path = folder / filename
                actual_hash = digest(path)
                recorded_hash = state.get('files', {}).get(filename)
                if recorded_hash is not None:
                    assert actual_hash == recorded_hash, f'Changed source: {path}'
                sources[str(path)] = actual_hash
    rows = []
    for task, metric, models in GROUPS:
        for display, name in models:
            pred, truth, metrics = data[name]
            row = [task, display]
            for time in TIMES:
                chunk = round(time / .32)
                assert abs(chunk * .32 - time) < 1e-10
                output = np.asarray(pred[:, chunk - 1, :], dtype=np.float64)
                assert np.isfinite(output).all()
                if metric == 'azimuth':
                    p = output[:, 2:4] if output.shape[1] == 4 else output
                    y = truth[:, 2:4]
                    angle = np.degrees(np.arctan2(p[:, 0], p[:, 1])) % 360
                    target = np.degrees(np.arctan2(y[:, 0], y[:, 1])) % 360
                    mae = float(np.abs((angle - target + 180) % 360 - 180).mean())
                    # Joint (pooled / variance-weighted) R² of the STORED sin/cos
                    # predictions. Do not normalize vectors or score raw degrees.
                    sse = np.square(p - y).sum(dtype=np.float64)
                    sst = np.square(y - y.mean(axis=0)).sum(dtype=np.float64)
                    score = float(1 - sse / sst) if sst > 0 else float('nan')
                    reference = r2_score(y, p, multioutput='variance_weighted', force_finite=False)
                else:
                    component = 0 if metric == 'magnitude' else 1
                    y = truth[:, component]
                    p = output[:, component] if output.shape[1] == 4 else output[:, 0]
                    mae = float(np.abs(p - y).mean())
                    sse = np.square(p - y).sum(dtype=np.float64)
                    sst = np.square(y - y.mean()).sum(dtype=np.float64)
                    score = float(1 - sse / sst) if sst > 0 else float('nan')
                    reference = r2_score(y, p, force_finite=False)
                assert np.isfinite(mae) and np.isfinite(score)
                assert abs(score - reference) < 1e-12
                existing = metrics[np.isclose(metrics.observation_time_s, time)]
                assert len(existing) == 1
                old_mae = float(existing.iloc[0][metric + '_mae'])
                assert abs(mae - old_mae) < 1e-10, (task, name, time, mae, old_mae)
                checks.append(dict(model=name, task=task, time_s=time, chunk=chunk,
                                   records=len(y), mae=mae, r2=score, mae_matches_saved=True,
                                   r2_matches_sklearn=True))
                row += [mae, score]
            rows.append(row)
    for path, expected in sources.items():
        assert digest(Path(path)) == expected
    payload = dict(headers=HEADERS, rows=rows, test_records=51772,
                   actual_times_s=list(TIMES), output_chunks=[9, 93],
                   azimuth_r2_definition='1 - SSE(sin,cos) / pooled SST(sin,cos); variance-weighted multioutput R2; stored clipped components, no vector normalization',
                   source_sha256=sources, checks=checks,
                   training_or_inference_performed=False)
    (HERE / 'test_table_values.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(pd.DataFrame(rows, columns=HEADERS).to_string(index=False, float_format=lambda x: f'{x:.8f}').replace('²', '^2'))
    print('Verified 18 task/model/time combinations; source hashes unchanged.')


if __name__ == '__main__':
    main()
