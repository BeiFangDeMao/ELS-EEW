"""Causal two-station replay: mean magnitude, distance-circle intersection and azimuth selection."""
import json
import math
from datetime import datetime
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.optimize import least_squares
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from infer_first_compactgru import load_model,clamp,SOURCE,WEIGHTS,CACHE
from plot_first_predictions import location
R=6371.


def vector(point):
    a,b=np.radians(point)
    return np.array([np.cos(a)*np.cos(b),np.cos(a)*np.sin(b),np.sin(a)])


def point(v):
    v=v/np.linalg.norm(v)
    return np.degrees([np.arcsin(np.clip(v[2],-1,1)),np.arctan2(v[1],v[0])])


def distance(a,b):
    return R*np.arctan2(np.linalg.norm(np.cross(vector(a),vector(b))),np.dot(vector(a),vector(b)))


def bearing(a,b):
    la,lo=np.radians(a); lb,lob=np.radians(b); dl=lob-lo
    return np.degrees(np.arctan2(np.sin(dl)*np.cos(lb),np.cos(la)*np.sin(lb)-np.sin(la)*np.cos(lb)*np.cos(dl)))%360


def angle_error(a,b):
    return abs((a-b+180)%360-180)


def solve(stations,radii,azimuths):
    u,v=map(vector,stations); dot=np.dot(u,v); determinant=1-dot*dot
    if determinant<1e-12: return None,'degenerate',None
    c1,c2=np.cos(np.array(radii)/R)
    base=((c1-dot*c2)*u+(c2-dot*c1)*v)/determinant
    height=1-np.dot(base,base)
    def score(p): return sum(angle_error(bearing(s,p),a) for s,a in zip(stations,azimuths))
    if height>=-1e-10:
        n=np.cross(u,v); n/=np.linalg.norm(n)
        candidates=[point(base+sign*np.sqrt(max(height,0))*n) for sign in (-1,1)]
        scores=[score(p) for p in candidates]
        return candidates[int(np.argmin(scores))],'intersecting',abs(scores[0]-scores[1])
    # Fixed fallback: minimize two spherical distance residuals, break equal fits by azimuth.
    starts=[location(*s,r,a) for s,r,a in zip(stations,radii,azimuths)]
    fits=[]
    for start in starts:
        fit=least_squares(lambda p:np.array([distance(s,p)-r for s,r in zip(stations,radii)]),
                          start,bounds=([-89.99,-540],[89.99,540]),max_nfev=100)
        if fit.success: fits.append((np.sum(fit.fun**2),score(fit.x),fit.x))
    if not fits: return None,'failed',None
    best=min(x[0] for x in fits)
    fit=min((x for x in fits if x[0]<=best+1e-6),key=lambda x:x[1])
    return fit[2],'nonintersecting_distance_fit',None


def fit_three(stations,radii,azimuths,with_angles):
    starts=[location(*s,r,a) for s,r,a in zip(stations,radii,azimuths)]
    for i,j in ((0,1),(0,2),(1,2)):
        candidate,_,_=solve([stations[i],stations[j]],[radii[i],radii[j]],[azimuths[i],azimuths[j]])
        if candidate is not None: starts.append(candidate)
    def residual(x):
        result=[(distance(s,x)-r)/10. for s,r in zip(stations,radii)]
        if with_angles:
            result += [((bearing(s,x)-a+180)%360-180)/30. for s,a in zip(stations,azimuths)]
        return result
    fits=[]
    for start in starts:
        f=least_squares(residual,start,bounds=([-89.99,-540],[89.99,540]),loss='huber',f_scale=1.,max_nfev=200)
        if f.success and np.isfinite(f.cost): fits.append(f)
    if not fits: return None
    return min(fits,key=lambda f:f.cost).x


def main():
    selection=HERE/'results/20261022_230231_957410/first_stations.json'
    events=json.loads(selection.read_text(encoding='utf-8'))['events']
    records=[r for e in events for r in e['selected'][:3]]
    waves=[]; offsets=[]
    with np.load(CACHE,allow_pickle=False) as c: test=set(c['keys'][c['splits']=='test'])
    with h5py.File(SOURCE,'r') as f:
        for row in records:
            if row['record'] not in test: raise ValueError(row['record'])
            ds=f[row['record']];w=ds[:].astype(np.float32).T
            if w.shape!=(3,2976) or not np.isfinite(w).all(): raise ValueError(row['record'])
            waves.append(w);offsets.append(float(ds.attrs.get('p_alignment_offset_s',0)))
    model,_,_=load_model();model=model.cpu();state=None;parts=[]
    x=torch.from_numpy(np.stack(waves))
    with torch.inference_mode():
        for chunk in x.split(32,-1):
            y,state=model.stream_step(chunk,state);parts.append(y)
        raw=torch.cat(parts,1)
        torch.testing.assert_close(raw,model(x)[0],rtol=2e-4,atol=2e-5)
    preds=clamp(raw.numpy())
    out=HERE/'results'/('three_station_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir()
    np.save(out/'station_predictions.npy',preds)
    rows=[];summaries=[]
    plt.rcParams.update({'font.family':['SimSun','DejaVu Sans'],'axes.unicode_minus':False})
    for i,event in enumerate(events):
        selected=event['selected'][:3];h=[r['header'] for r in selected]
        stations=[(v['station_latitude'],v['station_longitude']) for v in h]
        truth=(h[0]['latitude'],h[0]['longitude'])
        delays=[(datetime.fromisoformat(r['trigger_time'])-datetime.fromisoformat(selected[0]['trigger_time'])).total_seconds() for r in selected]
        local=[];active=None
        for j in range(93):
            t=(j+1)*.32+offsets[3*i]
            indices=[math.floor((t-d-offsets[3*i+n]+1e-9)/.32)-1 for n,d in enumerate(delays)]
            a=preds[3*i,j];az=float(np.degrees(np.arctan2(a[2],a[3]))%360)
            single=location(*stations[0],a[1],az)
            row=dict(event=event['name'],date=event['date'],time_since_first_p_s=t,
                first_magnitude=float(a[0]),true_magnitude=h[0]['magnitude'],
                first_error_km=distance(single,truth),three_magnitude=None,
                distance_only_error_km=None,distance_azimuth_error_km=None)
            if all(0<=k<93 and (k+1)*.32>=4. for k in indices[1:]):
                if active is None:active=t
                values=[preds[3*i+n,k] for n,k in enumerate(indices)]
                radii=[float(v[1]) for v in values]
                azimuths=[float(np.degrees(np.arctan2(v[2],v[3]))%360) for v in values]
                row['three_magnitude']=float(np.mean([v[0] for v in values]))
                for with_angles,name in [(False,'distance_only'),(True,'distance_azimuth')]:
                    pos=fit_three(stations,radii,azimuths,with_angles)
                    row[name+'_error_km']=distance(pos,truth) if pos is not None else None
                    row[name+'_latitude']=float(pos[0]) if pos is not None else None
                    row[name+'_longitude']=float(pos[1]) if pos is not None else None
            local.append(row);rows.append(row)
        d=pd.DataFrame(local)
        fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True,layout='constrained')
        t=d.time_since_first_p_s
        axes[0].plot(t,d.first_magnitude,label='First station',color='#888888')
        axes[0].plot(t,d.three_magnitude,label='Three-station mean',color='#2166ac')
        axes[0].axhline(h[0]['magnitude'],color='#d6604d',ls='--',label='True magnitude')
        axes[0].set_ylabel('Magnitude')
        axes[1].plot(t,d.first_error_km,label='First station',color='#888888')
        axes[1].plot(t,d.distance_only_error_km,label='Distance only',color='#2166ac')
        axes[1].plot(t,d.distance_azimuth_error_km,label='Distance + azimuth',color='#e69f00')
        axes[1].set(ylabel='Location error / km',xlabel='Time after first P (s)')
        for ax in axes:
            if active is not None: ax.axvline(active,color='#2a9d8f',ls='--',label='Three-station fusion starts')
            ax.grid(alpha=.2);ax.legend(fontsize=9)
        fig.suptitle(event['date']+' '+event['name']+' | '+' + '.join(r['station'] for r in selected))
        fig.savefig(out/(event['date']+'.png'),dpi=180);fig.savefig(out/(event['date']+'.pdf'));plt.close(fig)
        summaries.append(dict(event=event['name'],stations=[r['station'] for r in selected],delays_s=delays,
            fusion_start_s=active,final=local[-1]))
    pd.DataFrame(rows).to_csv(out/'replay.csv',index=False,encoding='utf-8-sig')
    (out/'results.json').write_text(json.dumps(dict(protocol='Three-station simultaneous absolute-time replay; >=4s input for each new station; original causal H5 and CompactGRU; CPU FP32; multi-start spherical Huber fitting; no truth in fitting',
        scales=dict(distance_km=10,azimuth_deg=30,huber_delta=1),
        limitation='Exploratory fixed scales, not validation calibrated; six case studies, no general performance claim. Distance-only initial seeds include predicted azimuth locations; objective uses only distance.',
        source=str(SOURCE),weights=str(WEIGHTS),selection=str(selection),events=summaries),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(out,flush=True)


if __name__=='__main__':main()
