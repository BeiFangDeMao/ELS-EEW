"""Independent Noto first-station replay with user-specified P shift +5.5s."""
import json
import math
import importlib.util
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_first_waveforms import read_wave
from infer_first_compactgru import HERE,ROOT,load_model,clamp,WEIGHTS
from plot_first_predictions import location


def main():
    selection=HERE/'results/20261022_230231_957410/first_stations.json'
    event=next(e for e in json.loads(selection.read_text(encoding='utf-8'))['events'] if e['date']=='20240101')
    record=event['selected'][0];h=record['header'];corrected=record['p_arrival_s']+5.5
    waves=[];rates=[]
    for ext in ('.UD','.NS','.EW'):
        t,w,fs=read_wave(record['saved_files'][ext]['path']);waves.append(w);rates.append(fs)
    if len(set(rates))!=1 or len(set(map(len,waves)))!=1:raise ValueError('分量不一致')
    fs=rates[0];stride=round(fs/100)
    if fs not in (100,200):raise ValueError('只支持100/200Hz原始记录')
    start=math.ceil(corrected*fs-1e-9);end=start+2976*stride
    if any(len(w)<end for w in waves):raise ValueError('修正后不足29.76秒，不补零')
    raw=np.stack([w[start:end:stride].astype(np.float32) for w in waves])
    path=ROOT/'els_eew/training/causal_preprocessing.py'
    spec=importlib.util.spec_from_file_location('_corrected_preprocessor',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    processor=module.CausalWavePreprocessor()
    processed=np.concatenate([processor.step(raw[None,:,j:j+32]) for j in range(0,2976,32)],axis=-1)
    model,_,_=load_model();model=model.cpu();state=None;parts=[]
    with torch.inference_mode():
        x=torch.from_numpy(processed)
        for chunk in x.split(32,-1):
            y,state=model.stream_step(chunk,state);parts.append(y)
        predictions=torch.cat(parts,1)
        torch.testing.assert_close(predictions,model(x)[0],rtol=2e-4,atol=2e-5)
    pred=clamp(predictions.numpy())[0]
    az=np.degrees(np.arctan2(pred[:,2],pred[:,3]))%360
    lat,lon=location(h['station_latitude'],h['station_longitude'],pred[:,1],az)
    a=np.sin((np.radians(lat)-math.radians(h['latitude']))/2)**2+np.cos(np.radians(lat))*math.cos(math.radians(h['latitude']))*np.sin((np.radians(lon)-math.radians(h['longitude']))/2)**2
    error=12742*np.arcsin(np.sqrt(np.clip(a,0,1)))
    # Match the project's label definitions, including unrounded station azimuth.
    with __import__('h5py').File(__import__('config').SOURCE,'r') as f:
        truth=[float(f[record['record']].attrs[k]) for k in ('mag','dis','azi')]
    out=HERE/'results'/('corrected_noto_first_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir()
    np.save(out/'processed_waveform.npy',processed)
    np.save(out/'predictions.npy',pred)
    times=np.arange(1,94)*.32+(start/fs-corrected)
    pd.DataFrame(dict(time_after_corrected_p_s=times,magnitude=pred[:,0],distance_km=pred[:,1],azimuth_deg=az,
        predicted_latitude=lat,predicted_longitude=lon,location_error_km=error)).to_csv(out/'predictions.csv',index=False)
    plt.rcParams.update({'font.family':['SimSun','DejaVu Sans'],'axes.unicode_minus':False})
    fig=plt.figure(figsize=(13,12),layout='constrained');grid=fig.add_gridspec(5,2,height_ratios=[.8,.8,.8,1.4,1.4])
    for i,w in enumerate(waves):
        ax=fig.add_subplot(grid[i,:]);t=np.arange(len(w))/fs-corrected;mask=(t>=-10)&(t<=30)
        ax.plot(t[mask],w[mask],lw=.6,color='#24384c');ax.axvline(0,color='red',ls='--',label='修正P到时')
        ax.axvline(-5.5,color='#888888',ls=':',label='原P到时');ax.set(xlim=(-10,30),ylabel=['Z / gal','N / gal','E / gal'][i]);ax.grid(alpha=.2)
        if i==0:ax.legend()
    for i,(y,title,ylabel,true) in enumerate([(pred[:,0],'震级估计','震级',truth[0]),(pred[:,1],'震中距估计','距离 / km',truth[1]),(az,'方位角估计','方位角 / °',truth[2]),(error,'震中定位误差','误差 / km',None)]):
        ax=fig.add_subplot(grid[3+i//2,i%2]);ax.plot(times,y,label='修正起点预测');ax.set(title=title,ylabel=ylabel,xlabel='修正P到时后 / s',xlim=(0,30));ax.grid(alpha=.2)
        if true is not None:ax.axhline(true,color='#d6604d',ls='--',label=f'真实值 {true:.2f}')
        ax.legend()
    fig.suptitle(f"能登半岛首台 {record['station']}｜修正图：P到时后移5.5秒",fontsize=14)
    fig.savefig(out/'Noto_first_corrected.png',dpi=180);fig.savefig(out/'Noto_first_corrected.pdf');plt.close(fig)
    (out/'manifest.json').write_text(json.dumps(dict(record=record['record'],station=record['station'],
        original_p_seconds=record['p_arrival_s'],corrected_p_seconds=corrected,shift_s=5.5,
        correction_source='user-specified; not an independently verified pick',source_selection=str(selection),
        source_files=record['saved_files'],weights=str(WEIGHTS),crop_start_native_sample=start,
        native_rate_hz=fs,output_rate_hz=100,preprocessing=processor.describe(),
        note='New raw crop then reset original causal preprocessing; old results unchanged; CPU FP32 stream consistency passed',
        truth=truth),ensure_ascii=False,indent=2),encoding='utf-8')
    print(out)


if __name__=='__main__':main()
