"""Self-contained six-event/top-three-station raw files and inference JSON bundle.

No training; exact original P picks. Original causal H5 supplies model inputs.
Only the selected 54 raw component files are copied, never archive packages.
"""
from datetime import datetime,timedelta
import hashlib
import json
from pathlib import Path
import shutil
import sys

import h5py
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(HERE/'plot_dependencies'))
from geographiclib.geodesic import Geodesic
from infer_first_compactgru import load_model,clamp,SOURCE,WEIGHTS,CACHE
from plot_first_waveforms import read_wave


def write_json(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    selection=HERE/'results/20261022_230231_957410/first_stations.json'
    report=json.loads(selection.read_text(encoding='utf-8'))
    events=report['events']
    records=[row for e in events for row in e['selected'][:3]]
    assert len(records)==18
    with np.load(CACHE,allow_pickle=False) as cache:
        test_keys=set(cache['keys'][cache['splits']=='test'])
    waves=[]; truths=[]; offsets=[]
    with h5py.File(SOURCE,'r') as source:
        for row in records:
            key=row['record']
            if key not in test_keys:raise ValueError(f'Not in original test set: {key}')
            ds=source[key];wave=ds[:].astype(np.float32).T
            if wave.shape!=(3,2976) or not np.isfinite(wave).all():raise ValueError(key)
            waves.append(wave)
            truths.append([float(ds.attrs[n]) for n in ('mag','dis','azi')])
            offsets.append(float(ds.attrs.get('p_alignment_offset_s',0)))
    model,_,checkpoint=load_model();model=model.cpu()
    x=torch.from_numpy(np.stack(waves));state=None;pieces=[]
    with torch.inference_mode():
        for chunk in x.split(32,-1):
            y,state=model.stream_step(chunk,state);pieces.append(y)
        raw=torch.cat(pieces,1)
        torch.testing.assert_close(raw,model(x)[0],rtol=2e-4,atol=2e-5)
    raw=raw.numpy();processed=clamp(raw)
    out=HERE/'prepared_data'/('Six_Events_Top3_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    out.mkdir(parents=True,exist_ok=False)
    summaries=[];picks=[];physical_peak=0.;source_files={}
    titles=['2024 Noto Peninsula','2024 Hyuganada','2024 Bungo Channel','2023 Noto','2024 Fukushima Offshore','2023 Southern Chiba']
    for i,event in enumerate(events):
        event_dir=out/event['date'];event_dir.mkdir()
        stations=[]
        for j,row in enumerate(event['selected'][:3]):
            n=3*i+j;folder=event_dir/f"station_{j+1}_{row['station']}";raw_dir=folder/'raw';raw_dir.mkdir(parents=True)
            files={};waveforms={};times=[];rates=[]
            for ext,label in zip(('.UD','.NS','.EW'),'ZNE'):
                source=Path(row['saved_files'][ext]['path'])
                if sha(source)!=row['saved_files'][ext]['sha256']:raise ValueError(f'Raw file changed: {source}')
                target=raw_dir/source.name;shutil.copy2(source,target)
                if sha(target)!=sha(source):raise ValueError('Copy hash mismatch')
                files[label]=str(target.relative_to(out)).replace('\\','/')
                source_files[files[label]]=str(source)
                t,w,fs=read_wave(target);t=t-row['p_arrival_s'];mask=(t>=-3)&(t<=29.76)
                if t[0]>-3 or t[-1]<29.76:raise ValueError(f'Missing real pre/post P: {source}')
                waveforms[label+'_gal']=w[mask].tolist();times.append(t[mask]);rates.append(fs)
                if j==0 and label=='Z':physical_peak=max(physical_peak,float(np.max(np.abs(w[mask]))))
            for t in times[1:]:np.testing.assert_array_equal(t,times[0])
            write_json(folder/'waveform_plot.json',dict(time_after_p_s=times[0].tolist(),**waveforms,
                unit='gal = cm/s^2',sampling_rate_hz=rates[0],p_seconds_from_record_start=row['p_arrival_s'],
                processing='fixed scale only, no normalization/filtering/baseline removal; real recorded pre-P noise'))
            h=row['header'];outputs=[]
            for k,p in enumerate(processed[n]):
                azi=float(np.degrees(np.arctan2(p[2],p[3]))%360)
                destination=Geodesic.WGS84.Direct(h['station_latitude'],h['station_longitude'],azi,float(p[1])*1000)
                nominal=(k+1)*.32;exact=nominal+offsets[n]
                outputs.append(dict(output_index=k+1,nominal_time_after_p_s=nominal,exact_time_after_p_s=exact,
                    absolute_output_available_time=(datetime.fromisoformat(row['trigger_time'])+timedelta(seconds=exact)).isoformat(timespec='microseconds'),
                    magnitude=float(p[0]),epicentral_distance_km=float(p[1]),azimuth_sin=float(p[2]),azimuth_cos=float(p[3]),
                    station_to_epicenter_azimuth_deg=azi,predicted_latitude_deg=destination['lat2'],predicted_longitude_deg=destination['lon2'],
                    raw_model_output=[float(v) for v in raw[n,k]]))
            truth=dict(magnitude=truths[n][0],epicentral_distance_km=truths[n][1],station_to_epicenter_azimuth_deg=truths[n][2],
                epicenter_latitude_deg=h['latitude'],epicenter_longitude_deg=h['longitude'])
            metadata=dict(record_id=row['record'],station_code=row['station'],rank=j+1,
                latitude_deg=h['station_latitude'],longitude_deg=h['station_longitude'],header=h,
                record_start_local_time=h['record_time'],p_pick_seconds=row['p_arrival_s'],absolute_p_time=row['trigger_time'],
                crop_alignment_offset_s=offsets[n],p_version='original spreadsheet; no correction',raw_files=files)
            write_json(folder/'inference.json',dict(metadata=metadata,truth=truth,
                model=dict(name='CompactGRU',checkpoint_sha256=sha(WEIGHTS),epoch=int(checkpoint['epoch']),
                    input_shape=[3,2976],sampling_rate_hz=100,chunk_samples=32,output_count=93,
                    input='Original causal H5 waveform, no added preprocessing',
                    preprocessing=dict(sampling_rate_hz=100,band_hz=[.75,40],butter_design_order=4,
                        total_bandpass_order=8,subtract_first_sample=True,filter='one-sided SOS',
                        normalization=False,linear_detrend=False,zero_phase_filter=False),
                    clipping=dict(magnitude=[0,10],distance_km=[0,200],sin_cos=[-1,1]),
                    raw_output_order=['magnitude','distance_km','sin_azimuth','cos_azimuth']),
                geometry='WGS84 direct; azimuth clockwise from geographic north, station toward epicenter',outputs=outputs))
            stations.append(dict(rank=j+1,station_code=row['station'],record_id=row['record'],
                inference_json=str((folder/'inference.json').relative_to(out)).replace('\\','/'),
                waveform_json=str((folder/'waveform_plot.json').relative_to(out)).replace('\\','/'),raw_files=files))
            picks.append(dict(event_date=event['date'],record_id=row['record'],station_code=row['station'],
                              rank=j+1,p_seconds=row['p_arrival_s'],absolute_p_time=row['trigger_time']))
        summaries.append(dict(event_order=i+1,date=event['date'],name=event['name'],english_name=titles[i],
                              title=f'({chr(97+i)}) {titles[i]} — {stations[0]["station_code"]}',stations=stations))
    write_json(out/'events.json',summaries)
    write_json(out/'original_selected_picks.json',picks)
    previous=ROOT/'plotting/Plotting_Data_Packages'
    configs=sorted(previous.glob('Single_Station_6Events_*/plot_settings.json'))
    if configs:
        settings=json.loads(configs[-1].read_text(encoding='utf-8'))
        settings['waveform']['ylim_gal']=[-float(np.ceil(physical_peak/100)*100),float(np.ceil(physical_peak/100)*100)]
        settings['data_entry']='events.json -> stations[0] for single-station plots; stations[1:3] remain independent station results'
        write_json(out/'plot_settings.json',settings)
    map_dir=out/'map_data';map_dir.mkdir()
    shutil.copy2(ROOT/'plotting/map_data/ne_10m_land.geojson',map_dir/'ne_10m_land.geojson')
    guide=HERE/'BUNDLE_GUIDE_TEMPLATE.md'
    if guide.exists():
        (out/'README_绘图说明.md').write_text(guide.read_text(encoding='utf-8').replace(
            'provenance_manifest.json','manifest.json').replace('YYYYMMDD_event/','YYYYMMDD/'),encoding='utf-8')
    shutil.copy2(__file__,out/Path(__file__).name)
    files={str(p.relative_to(out)).replace('\\','/'):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
    write_json(out/'manifest.json',dict(events=6,stations=18,raw_components=54,outputs_per_station=93,
        selection_source=str(selection),source_h5=str(SOURCE),checkpoint=str(WEIGHTS),checkpoint_sha256=sha(WEIGHTS),
        selection_policy='original chronological top3 of P-matched K-NET triplets; original ISK00313.52; no exclusion or swap',
        inference='CPU FP32 original causal stream; full/stream check passed; no training; original test records only',
        absolute_time='header local time (Asia/Tokyo); Record Time + original annotated P seconds; no extra timing correction',
        source_raw_paths=source_files,files_sha256=files))
    # Validate all 18 JSON documents and raw triplets, not just their existence.
    for event in summaries:
        for station in event['stations']:
            d=json.loads((out/station['inference_json']).read_text(encoding='utf-8'))
            assert len(d['outputs'])==93
            assert all((out/p).is_file() for p in station['raw_files'].values())
    (HERE/'prepared_data/latest_bundle.json').write_text(json.dumps(dict(directory=str(out)),indent=2),encoding='utf-8')
    print(out,flush=True)


if __name__=='__main__':main()
