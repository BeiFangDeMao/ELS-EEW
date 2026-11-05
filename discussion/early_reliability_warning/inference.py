"""Original CompactGRU inference, metadata audit, and full Parquet export."""
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm

from config import *


def dump(path,data):
    def clean(x):
        if isinstance(x,dict): return {str(k):clean(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)): return [clean(v) for v in x]
        if isinstance(x,np.generic): return clean(x.item())
        if isinstance(x,float) and not np.isfinite(x): return None
        if isinstance(x,Path): return str(x)
        return x
    path.write_text(json.dumps(clean(data),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def separation(lat1,lon1,lat2,lon2):
    a = np.sin((lat2-lat1)/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
    return 12742*np.arcsin(np.sqrt(np.clip(a,0,1)))


def destination(lat,lon,distance,angle):
    delta = distance/6371
    la = np.arcsin(np.clip(np.sin(lat)*np.cos(delta)+np.cos(lat)*np.sin(delta)*np.cos(angle),-1,1))
    lo = lon+np.arctan2(np.sin(angle)*np.sin(delta)*np.cos(lat),np.cos(delta)-np.sin(lat)*np.sin(la))
    return la,(lo+np.pi)%(2*np.pi)-np.pi


def scalar(value):
    if isinstance(value,bytes): return value.decode('utf-8')
    if isinstance(value,np.generic): return value.item()
    return value


class TestData(Dataset):
    def __init__(self,keys):
        self.keys = keys
        self.file = None

    def __len__(self): return len(self.keys)

    def __getitem__(self,index):
        if self.file is None: self.file = h5py.File(SOURCE,'r')
        key = str(self.keys[index])
        ds = self.file[key]
        wave = np.ascontiguousarray(ds[:].astype(np.float32).T)
        attrs = {k:scalar(v) for k,v in ds.attrs.items()}
        assert wave.shape==(3,LENGTH) and np.isfinite(wave).all()
        truth = np.array([attrs[k] for k in ('mag','dis','azi','station_lat','station_lon','lat','lon')],dtype=np.float64)
        assert np.isfinite(truth).all() and truth[0]>=3 and 0<=truth[1]<200
        assert attrs['sampling_freq_hz']==100 and attrs['channel_order']=='Z,N,E' and attrs['waveform_unit']=='gal'
        fields = ['source_dataset','origin_time','record_time','pt_sec','p_arrival_relative_s','p_alignment_offset_s',
                  'depth_km','modified_pat_sample','original_sampling_freq_hz','crop_start']
        meta = dict(record_id=key,**{k:attrs.get(k,None) for k in fields})
        return wave,truth,json.dumps(meta,ensure_ascii=False)


def selected_keys():
    provenance = json.loads((CACHE.parent/'data_provenance.json').read_text(encoding='utf-8'))
    stat = SOURCE.stat()
    assert stat.st_size==provenance['source_size'] and stat.st_mtime_ns==provenance['source_mtime_ns']
    with np.load(CACHE,allow_pickle=False) as data:
        mask = data['splits']=='test'
        keys,labels = data['keys'][mask],data['labels'][mask]
    assert len(keys)==51772 and all(k[6:8] in ('23','24') for k in keys)
    assert len(set(keys))==len(keys)
    return keys,labels


def load_model():
    sys.path.insert(0,str(PROJECT))
    from models.compact_gru import CompactGRU
    checkpoint = torch.load(WEIGHTS,map_location='cpu',weights_only=False)
    assert checkpoint['model_id']=='CompactGRU' and checkpoint['sequence_length']==LENGTH
    model = CompactGRU()
    model.load_state_dict(checkpoint['model_state_dict'],strict=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(4)
    return model.to(device).eval(),device,checkpoint


def clamp(pred):
    out = pred.copy()
    out[:,:,0] = np.clip(out[:,:,0],0,10)
    out[:,:,1] = np.clip(out[:,:,1],0,200)
    out[:,:,2:] = np.clip(out[:,:,2:],-1,1)
    return out


def validate_small(out):
    keys,labels = selected_keys()
    indices = np.linspace(0,len(keys)-1,16,dtype=int)
    data = TestData(keys[indices])
    wave = torch.from_numpy(np.stack([data[i][0] for i in range(len(data))]))
    model,device,_ = load_model()
    wave = wave.to(device)
    with torch.inference_mode():
        full,_ = model(wave)
        state,parts = None,[]
        for start in range(0,LENGTH,CHUNK):
            part,state = model.stream_step(wave[:,:,start:start+CHUNK],state)
            parts.append(part)
        stream = torch.cat(parts,dim=1)
        assert torch.allclose(full,stream,atol=1e-4,rtol=2e-5), 'Full/stream mismatch'
        altered = wave.clone()
        altered[:,:,1024:] = torch.randn_like(altered[:,:,1024:])*100
        changed,_ = model(altered)
        assert torch.allclose(full[:,:32],changed[:,:32],atol=1e-5,rtol=1e-6), 'Future input affects prefix'
        prefix,_ = model(wave[:,:,:1024])
        assert torch.allclose(full[:,:32],prefix,atol=1e-4,rtol=2e-5)
    assert np.array_equal(np.stack([data[i][1] for i in range(len(data))])[:,:3],labels[indices])
    # Coordinate convention: true distance/azimuth must reconstruct catalog location to label rounding accuracy.
    m = np.stack([data[i][1] for i in range(len(data))])
    la,lo = destination(np.deg2rad(m[:,3]),np.deg2rad(m[:,4]),m[:,1],np.deg2rad(m[:,2]))
    assert np.max(separation(la,lo,np.deg2rad(m[:,5]),np.deg2rad(m[:,6])))<=.0051
    result = dict(records=16,indices=indices.tolist(),device=str(device),
        raw_max_abs_full_vs_stream=float((full-stream).abs().max()),
        per_channel_max_abs_full_vs_stream=(full-stream).abs().amax(dim=(0,1)).cpu().tolist(),
        future_suffix_invariance=True,prefix_invariance=True,labels_match=True,coordinate_direction_verified=True,
        inference_mode_selected='existing full-window causal forward, validated against stream_step')
    dump(out/'minimal_validation.json',result)
    print('MINIMAL_VALIDATION='+json.dumps(result),flush=True)
    if data.file is not None: data.file.close()
    return result


def infer(out):
    keys,cache_labels = selected_keys()
    model,device,checkpoint = load_model()
    n,t = len(keys),LENGTH//CHUNK
    pred_dir = out/'predictions'
    pred_dir.mkdir()
    raw_all = np.lib.format.open_memmap(pred_dir/'raw_predictions.npy',mode='w+',dtype=np.float32,shape=(n,t,4))
    processed_all = np.lib.format.open_memmap(pred_dir/'processed_predictions.npy',mode='w+',dtype=np.float32,shape=(n,t,4))
    err_all = np.lib.format.open_memmap(pred_dir/'location_errors_km.npy',mode='w+',dtype=np.float64,shape=(n,t))
    truths = np.empty((n,7),dtype=np.float64)
    rows = []
    loader = DataLoader(TestData(keys),batch_size=BATCH_SIZE,shuffle=False,num_workers=WORKERS,pin_memory=device.type=='cuda')
    offset = 0
    with torch.inference_mode():
        for wave,truth,metadata in tqdm(loader,desc='Full independent test replay',mininterval=5):
            raw,_ = model(wave.to(device,non_blocking=True))
            assert torch.isfinite(raw).all(), 'Do not replace nonfinite predictions'
            raw = raw.cpu().numpy()
            pred = clamp(raw)
            truth = truth.numpy()
            end = offset+len(truth)
            assert np.array_equal(truth[:,:3],cache_labels[offset:end])
            slat,slon,tlat,tlon = [np.deg2rad(truth[:,i:i+1]) for i in range(3,7)]
            angle = np.arctan2(pred[:,:,2].astype(np.float64),pred[:,:,3].astype(np.float64))
            plat,plon = destination(slat,slon,pred[:,:,1].astype(np.float64),angle)
            err = separation(plat,plon,tlat,tlon)
            raw_all[offset:end],processed_all[offset:end],err_all[offset:end] = raw,pred,err
            truths[offset:end] = truth
            rows.extend(json.loads(x) for x in metadata)
            offset = end
    assert offset==n
    raw_all.flush(); processed_all.flush(); err_all.flush()
    np.save(pred_dir/'truths.npy',truths)
    meta = pd.DataFrame(rows)
    for i,name in enumerate(['true_magnitude','true_distance_km','true_azimuth_deg','station_lat','station_lon','event_lat','event_lon']):
        meta[name] = truths[:,i]
    meta,audit = audit_metadata(meta)
    meta.to_parquet(pred_dir/'records.parquet',index=False)
    meta.to_csv(out/'record_metadata.csv',index=False,encoding='utf-8-sig')
    dump(out/'metadata_audit.json',audit)
    original = json.loads((CHECKPOINT_DIR/'test_metrics_by_chunk.json').read_text(encoding='utf-8'))
    replay_checks = {}
    for col,name in enumerate(('mag','dis')):
        value = np.mean(abs(processed_all[:,:,col].astype(np.float64)-truths[:,col:col+1]),axis=0)
        reference = np.array([x['metrics'][name]['mae'] for x in original['by_chunk']])
        difference = float(np.max(abs(value-reference)))
        assert difference<.002
        replay_checks[name+'_mae_max_difference'] = difference
    reference_location = EXPERIMENT/'localization_retest_20261016_163704_001976/location_errors_km.npy'
    if reference_location.exists():
        previous = np.load(reference_location,mmap_mode='r')
        difference = float(np.max(abs(err_all-previous)))
        assert difference<.002
        replay_checks['location_max_difference_km'] = difference
    manifest = dict(model='CompactGRU',weights=str(WEIGHTS),weight_sha256=hashlib.sha256(WEIGHTS.read_bytes()).hexdigest(),
        checkpoint_epoch=int(checkpoint['epoch']),test_records=n,num_outputs=t,source_h5=str(SOURCE),
        test_key_sha256=hashlib.sha256('\n'.join(keys).encode()).hexdigest(),
        sampling_rate_hz=RATE,chunk_samples=CHUNK,nominal_output_times_s=((np.arange(t)+1)*CHUNK/RATE).tolist(),
        input='B,3,2976, Z/N/E, gal; existing stored causal preprocessing, no added transforms',
        raw_output_order=['magnitude','distance_km','sin_back_azimuth','cos_back_azimuth'],
        clipping={'magnitude':[0,10],'distance_km':[0,200],'sin_back_azimuth':[-1,1],'cos_back_azimuth':[-1,1]},
        location='Spherical forward destination from station, then great-circle separation from catalog, R=6371 km',
        time='Nominal post-aligned-P waveform duration; exact post-annotated-P time adds p_alignment_offset_s (saved per prediction).',
        observed_p_alignment_offset_range_s=[float(meta.p_alignment_offset_s.min()),float(meta.p_alignment_offset_s.max())],
        replay_checks=replay_checks,inference_completed=True,
        metadata_limits=audit['limitations'])
    dump(out/'inference_manifest.json',manifest)
    export_predictions(out,meta,truths,raw_all,processed_all,err_all)
    return meta


def audit_metadata(meta):
    """Single-waveform evaluation only; never infer event or station identities."""
    excluded = ['event_id','earthquake_id','eventid','source_id','provided_event_id','event_group_id',
                'event_id_status','station_id','station_code','origin_datetime','seconds_origin_to_p',
                'absolute_clock_suspect','event_group_clock_suspect','source_priority',
                'is_proxy_first_available_station','is_reliable_event_first_station']
    meta = meta.drop(columns=excluded,errors='ignore').copy()
    meta['p_arrival_absolute'] = pd.to_datetime(meta.record_time,errors='coerce')+pd.to_timedelta(pd.to_numeric(meta.pt_sec,errors='coerce'),unit='s')
    offsets = pd.to_numeric(meta.p_alignment_offset_s,errors='coerce')
    assert offsets.notna().all() and offsets.between(-1e-7,.010001).all()
    limitations = [
        'Only earthquake waveforms selected with M>=3 and D<200 km; no continuous noise or non-earthquake records.',
        'P-aligned retrospective replay using annotated arrivals; no automatic P detection, telemetry, or end-to-end alert lead-time evaluation.',
        'All analysis units are individual waveform records, as requested. No event grouping, station identification, or first-station selection.',
        'Header absolute times are retained for provenance only; decisions use per-waveform post-P observation time, not origin-time or cross-station comparisons.',
    ]
    audit = dict(analysis_unit='waveform_record',records=len(meta),event_or_station_ids_computed=False,
        missing_absolute_arrival_times=int(meta.p_arrival_absolute.isna().sum()),
        limitations=limitations,
        truth_ranges={k:[meta[k].min(),meta[k].max()] for k in ('true_magnitude','true_distance_km','true_azimuth_deg')})
    return meta,audit


def export_predictions(out,meta,truth,raw,pred,errors):
    n,t = errors.shape
    writer = None
    path = out/'predictions/predictions.parquet'
    metadata_fields = ['record_id','source_dataset',
                       'p_arrival_absolute','origin_time','record_time','pt_sec','p_alignment_offset_s',
                       'true_magnitude','true_distance_km','true_azimuth_deg','station_lat','station_lon','event_lat','event_lon']
    try:
        for start in tqdm(range(0,n,512),desc='Export prediction Parquet',mininterval=5):
            end = min(start+512,n)
            index = np.repeat(np.arange(start,end),t)
            df = meta.iloc[index][metadata_fields].reset_index(drop=True)
            times = np.tile((np.arange(t)+1)*CHUNK/RATE,end-start)
            df['chunk'] = np.tile(np.arange(1,t+1),end-start)
            df['nominal_observation_time_s'] = times
            df['observation_time_after_annotated_p_s'] = times+df.p_alignment_offset_s.to_numpy()
            p,r = np.asarray(pred[start:end]).reshape(-1,4),np.asarray(raw[start:end]).reshape(-1,4)
            for i,name in enumerate(('magnitude','distance_km','sin_azimuth','cos_azimuth')):
                df['raw_predicted_'+name] = r[:,i]
                df['predicted_'+name] = p[:,i]
            angle = np.arctan2(p[:,2].astype(np.float64),p[:,3].astype(np.float64))
            df['predicted_azimuth_deg'] = np.rad2deg(angle)%360
            la,lo = destination(np.deg2rad(df.station_lat.to_numpy()),np.deg2rad(df.station_lon.to_numpy()),p[:,1].astype(np.float64),angle)
            df['predicted_epicenter_lat'],df['predicted_epicenter_lon'] = np.rad2deg(la),np.rad2deg(lo)
            df['location_error_km'] = np.asarray(errors[start:end]).reshape(-1)
            df['magnitude_absolute_error'] = abs(df.predicted_magnitude-df.true_magnitude)
            df['distance_absolute_error_km'] = abs(df.predicted_distance_km-df.true_distance_km)
            df['azimuth_absolute_error_deg'] = abs((df.predicted_azimuth_deg-df.true_azimuth_deg+180)%360-180)
            table = pa.Table.from_pandas(df,preserve_index=False)
            if writer is None: writer = pq.ParquetWriter(path,table.schema,compression='zstd')
            writer.write_table(table)
    finally:
        if writer is not None: writer.close()
    assert pq.ParquetFile(path).metadata.num_rows==n*t
