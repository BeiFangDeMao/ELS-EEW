"""Uniform original postprocessing, spherical localization and stability."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from ablation_models import clamp_active,INDICES


def geometry_coordinates(cfg,keys,folder):
    import h5py
    coords=[]
    with h5py.File(cfg['dataset'],'r') as file:
        for key in tqdm(keys,desc='Read test geometry, no waveform reprocessing'):
            attrs=file[key].attrs
            coords.append([attrs[k] for k in ('station_lat','station_lon','lat','lon')])
    coords=np.asarray(coords,dtype=np.float64)
    if coords.shape!=(len(keys),4) or not np.isfinite(coords).all(): raise ValueError('Invalid coordinates')
    np.savez(folder/'test_geometry.npz',keys=np.asarray(keys),coordinates=coords)
    return coords


def evaluate_saved(pred,truth,spec,cfg,coords=None):
    # pred (N,T,active_output_dim), truth (N,4) as original labels.
    from inference import destination,separation
    pred=np.asarray(pred,dtype=np.float64);truth=np.asarray(truth,dtype=np.float64)
    n,t,_=pred.shape;times=np.arange(1,t+1)*.32
    task=spec['task'];errors={};changes={}
    if task in ('all','magnitude'):
        p=pred[:,:,0]
        errors['magnitude']=np.abs(p-truth[:,0:1]);changes['magnitude']=np.abs(np.diff(p,axis=1))
    if task in ('all','distance'):
        p=pred[:,:,1 if task=='all' else 0]
        errors['distance']=np.abs(p-truth[:,1:2]);changes['distance']=np.abs(np.diff(p,axis=1))
    if task in ('all','azimuth'):
        p=pred[:,:,2:4] if task=='all' else pred
        angle=np.degrees(np.arctan2(p[:,:,0],p[:,:,1]))%360
        true_angle=np.degrees(np.arctan2(truth[:,2:3],truth[:,3:4]))%360
        errors['azimuth']=np.abs((angle-true_angle+180)%360-180)
        changes['azimuth']=np.abs((np.diff(angle,axis=1)+180)%360-180)
    if task=='all' and coords is not None:
        la,lo,ela,elo=[np.radians(coords[:,i:i+1]) for i in range(4)]
        pla,plo=destination(la,lo,pred[:,:,1],np.radians(angle))
        errors['localization']=separation(ela,elo,pla,plo)
    metrics=[]
    for j,time in enumerate(times):
        row=dict(model=spec['name'],observation_time_s=float(time),records=n)
        for key,e in errors.items():
            row[f'{key}_mae']=float(e[:,j].mean())
            row[f'{key}_rmse']=float(np.sqrt(np.square(e[:,j]).mean()))
            row[f'{key}_median_ae']=float(np.median(e[:,j]))
        metrics.append(row)
    stability=[]
    for key,delta in changes.items():
        for window,(lower,upper) in cfg['stability_windows'].items():
            mask=(times[:-1]>=lower-1e-9)&(times[1:]<=upper+1e-9)
            d=delta[:,mask]
            stability.append(dict(model=spec['name'],task=key,window=window,records=n,
                transition_count_per_record=int(mask.sum()),first_from_time_s=float(times[:-1][mask][0]) if mask.any() else np.nan,
                last_to_time_s=float(times[1:][mask][-1]) if mask.any() else np.nan,
                mean_abs_change=float(d.mean()) if d.size else np.nan,median_abs_change=float(np.median(d)) if d.size else np.nan,
                p90_abs_change=float(np.percentile(d,90)) if d.size else np.nan,
                corresponding_prediction_mae=float(errors[key][:,1:][:,mask].mean()) if d.size else np.nan,
                unit='degrees per 0.32s update' if key=='azimuth' else 'km per 0.32s update' if key=='distance' else 'magnitude per 0.32s update'))
    return pd.DataFrame(metrics),pd.DataFrame(stability)


def evaluate_model(model,loader,spec,cfg,folder,coords,device):
    n=len(loader.dataset);dims=len(INDICES[spec['task']])
    prediction=np.lib.format.open_memmap(folder/'predictions.npy',mode='w+',dtype=np.float32,shape=(n,93,dims))
    truth=np.empty((n,4),dtype=np.float32);offset=0
    model.to(device).eval()
    with torch.inference_mode():
        for wave,y,_,_ in tqdm(loader,desc=f'{spec["name"]} unified test'):
            raw=model(wave.to(device))[0]
            if raw.shape!=(len(wave),93,dims) or not torch.isfinite(raw).all(): raise FloatingPointError('Bad inference output')
            prediction[offset:offset+len(wave)]=clamp_active(raw,spec['task']).cpu().numpy()
            truth[offset:offset+len(wave)]=y.numpy();offset+=len(wave)
    assert offset==n
    prediction.flush()
    np.save(folder/'truth_labels.npy',truth)
    metrics,stability=evaluate_saved(prediction,truth,spec,cfg,coords)
    metrics.to_csv(folder/'metrics_by_time.csv',index=False)
    stability.to_csv(folder/'stability.csv',index=False)
    return metrics,stability
