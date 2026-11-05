# -*- coding: utf-8 -*-
"""Fill missing model scales using saved best weights; preserve prior evaluations."""
from datetime import datetime
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
WORKSPACE = STUDY.parents[1]
sys.path.insert(0, str(WORKSPACE/'els_eew'))
sys.path.insert(0, str(STUDY))
sys.path.insert(0, str(WORKSPACE/'discussion/early_reliability_warning'))
from model_scaling import make_model, validate_streaming
from inference import TestData, selected_keys, destination, separation, clamp


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def main():
    active = read(STUDY/'active_run.json')
    run = Path(active['results_root'])/active['run_directory_name']
    old = Path(read(HERE/'latest_result.json')['result_directory'])
    old_manifest = read(old/'manifest.json')
    assert Path(old_manifest['source_experiment']).resolve() == run.resolve()
    keys, labels = selected_keys()
    np.testing.assert_array_equal(np.load(old/'record_keys.npy', allow_pickle=False), keys)
    old_truth = np.load(old/'truths.npy', mmap_mode='r')
    np.testing.assert_array_equal(old_truth[:, :3], labels)
    previous = pd.read_csv(old/'localization_by_time.csv')
    names = ['ELS-2K', 'ELS-4K', 'ELS-8K', 'ELS-16K', 'ELS-32K']
    missing = []
    for name in names:
        rows = previous[(previous.model == name) & (previous.group == 'all')]
        if rows.empty:
            missing.append(name)
        else:
            assert len(rows) == 93 and (rows.records == 51772).all()
            np.testing.assert_allclose(rows.observation_time_s, np.arange(1,94)*.32)
    if not missing:
        print('All five scales already evaluated:', old, flush=True)
        return
    out = HERE/'results'/(datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_all_five_scales')
    out.mkdir(parents=True)
    print('OUTPUT:', out, flush=True)
    print('Inference only for:', missing, '; reuse prior scales:', sorted(set(names)-set(missing)), flush=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(4)
    torch.manual_seed(42)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    sample = TestData(keys[:2])
    wave = torch.from_numpy(np.stack([sample[i][0] for i in range(2)])).to(device)
    if sample.file is not None:
        sample.file.close()
    models, provenance = {}, dict(old_manifest['models'])
    for name in missing:
        folder = run/name
        state = read(folder/'state.json')
        assert state['status'] == 'complete'
        for filename, expected in state['output_hashes'].items():
            assert digest(folder/filename) == expected, filename
        spec = read(folder/'config.json')['model']
        checkpoint = torch.load(folder/'best.pth', map_location='cpu', weights_only=False)
        model = make_model(spec).to(device).eval()
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        assert sum(p.numel() for p in model.parameters()) == spec['total_parameters']
        check = validate_streaming(model, wave)
        models[name] = model
        provenance[name] = dict(checkpoint=str(folder/'best.pth'), sha256=digest(folder/'best.pth'),
                                epoch=int(checkpoint['epoch']), stream_check=check)
        print(name, 'real-wave full/stream and causal prefix checks passed', flush=True)
    n, t = len(keys), 93
    predictions = {name:np.lib.format.open_memmap(out/f'{name}_predictions.npy', mode='w+', dtype=np.float32, shape=(n,t,4)) for name in missing}
    errors = {name:np.lib.format.open_memmap(out/f'{name}_location_errors_km.npy', mode='w+', dtype=np.float64, shape=(n,t)) for name in missing}
    truths = np.lib.format.open_memmap(out/'truths.npy', mode='w+', dtype=np.float64, shape=(n,7))
    np.save(out/'record_keys.npy', keys, allow_pickle=False)
    data = TestData(keys)
    loader = DataLoader(data, batch_size=512, shuffle=False, num_workers=0)
    offset = 0
    with torch.inference_mode():
        for x, y, _ in tqdm(loader, desc='Missing scales localization; identical 51,772 test records', mininterval=10):
            truth = y.numpy()
            end = offset+len(truth)
            sl = slice(offset,end)
            np.testing.assert_array_equal(truth, old_truth[sl])
            truths[sl] = truth
            lat, lon = np.deg2rad(truth[:,3:4]), np.deg2rad(truth[:,4:5])
            true_lat, true_lon = np.deg2rad(truth[:,5:6]), np.deg2rad(truth[:,6:7])
            x = x.to(device)
            for name, model in models.items():
                raw = model(x)[0]
                assert raw.shape == (len(truth),93,4) and torch.isfinite(raw).all()
                p = clamp(raw.cpu().numpy())
                predictions[name][sl] = p
                angle = np.arctan2(p[:,:,2].astype(np.float64), p[:,:,3].astype(np.float64))
                la, lo = destination(lat, lon, p[:,:,1].astype(np.float64), angle)
                error = separation(true_lat, true_lon, la, lo)
                assert np.isfinite(error).all() and (error >= 0).all()
                errors[name][sl] = error
            offset = end
    assert offset == n
    if data.file is not None:
        data.file.close()
    for array in [truths, *predictions.values(), *errors.values()]:
        array.flush()
    original = pd.read_csv(run/'accuracy_by_time.csv')
    checks = []
    for name, pred in predictions.items():
        for time in [2.88,4.8,9.92,19.84,29.76]:
            j = round(time/.32)-1
            angle = np.rad2deg(np.arctan2(pred[:,j,2].astype(np.float64), pred[:,j,3].astype(np.float64)))%360
            values = dict(magnitude_mae=float(abs(pred[:,j,0].astype(np.float64)-truths[:,0]).mean()),
                          distance_mae_km=float(abs(pred[:,j,1].astype(np.float64)-truths[:,1]).mean()),
                          azimuth_circular_mae_deg=float(abs((angle-truths[:,2]+180)%360-180).mean()))
            prior = original[(original.model == name) & (original.split == 'test') & np.isclose(original.actual_output_time_s, time)].iloc[0]
            differences = {key:values[key]-float(prior[key]) for key in values}
            assert all(abs(v) < 1e-3 for v in differences.values()), differences
            checks.append(dict(model=name, time_s=time, differences=differences))
    groups = {'all':np.ones(n,dtype=bool)}
    groups.update({f'D_le_{d}':truths[:,1] <= d for d in [10,50,100,150]})
    groups.update({f'D_{lo}_{hi}':(truths[:,1]>=lo)&(truths[:,1]<hi) for lo,hi in [(0,10),(10,50),(50,100),(100,200)]})
    rows = []
    for name, error in errors.items():
        for group, mask in groups.items():
            selected = np.asarray(error[mask])
            median, p90 = np.percentile(selected, [50,90], axis=0, method='linear')
            means = selected.mean(axis=0)
            rates = {v:(selected<=v).mean(axis=0) for v in [5,10,20,30,50]}
            for j in range(93):
                rows.append(dict(model=name, group=group, records=int(mask.sum()), observation_time_s=(j+1)*.32,
                    mean_location_error_km=float(means[j]),median_location_error_km=float(median[j]),p90_location_error_km=float(p90[j]),
                    **{f'within_{v}km':float(series[j]) for v,series in rates.items()}))
    merged = pd.concat([previous,pd.DataFrame(rows)],ignore_index=True)
    assert not merged.duplicated(['model','group','observation_time_s']).any()
    merged.to_csv(out/'localization_by_time.csv',index=False,encoding='utf-8-sig')
    key_times = merged[np.isclose(merged.observation_time_s,2.88)|np.isclose(merged.observation_time_s,29.76)]
    key_times.to_csv(out/'localization_key_times.csv',index=False,encoding='utf-8-sig')
    for name in missing:
        assert digest(provenance[name]['checkpoint']) == provenance[name]['sha256']
    manifest = dict(source_experiment=str(run), source_dataset=old_manifest['source_dataset'], records=n,
        aggregation='waveform_record',time_points=t,batch_size=512,spherical_radius_km=6371,
        geometry=old_manifest['geometry'],clipping=old_manifest['clipping'],models=provenance,
        training_performed=False,new_inference_models=missing,reused_result_directory=str(old),
        reused_metrics_sha256=digest(old/'localization_by_time.csv'),
        validation_checks_against_scaling_metrics=checks,
        truth_columns=old_manifest['truth_columns'],
        reused_error_arrays={name:str(old/f'{name}_location_errors_km.npy') for name in names if name not in missing},
        cuda_cudnn_tf32=torch.backends.cudnn.allow_tf32,completed=True)
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    report = ['# Completed five-scale localization evaluation', '',
        'Inference only for missing scales; existing 8K/16K/32K metrics reused unchanged. Same ordered 51,772 independent test records, all 93 times.', '',
        'Sphere radius 6371 km; station-to-epicenter predicted bearing and distance. No training or preprocessing changes. Saved predictions and errors for both new scales.', '',
        key_times[key_times.group=='all'][['model','observation_time_s','mean_location_error_km']].to_string(index=False)]
    (out/'summary.md').write_text('\n'.join(report),encoding='utf-8')
    (HERE/'latest_result.json').write_text(json.dumps(dict(result_directory=str(out)),indent=2),encoding='utf-8')
    print('COMPLETE:', out, flush=True)
    print(key_times[key_times.group=='all'][['model','observation_time_s','mean_location_error_km']].to_string(index=False),flush=True)


if __name__ == '__main__':
    main()
