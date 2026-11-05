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


def main():
    selection=HERE/'results/20261022_230231_957410/first_stations.json'
    events=json.loads(selection.read_text(encoding='utf-8'))['events']
    records=[r for e in events for r in e['selected'][:2]]
    waves=[]; offsets=[]
    with np.load(CACHE,allow_pickle=False) as c: test=set(c['keys'][c['splits']=='test'])
    with h5py.File(SOURCE,'r') as f:
        for row in records:
            if row['record'] not in test: raise ValueError(row['record'])
            ds=f[row['record']]; w=ds[:].astype(np.float32).T
            if w.shape!=(3,2976) or not np.isfinite(w).all(): raise ValueError(row['record'])
            waves.append(w); offsets.append(float(ds.attrs.get('p_alignment_offset_s',0)))
    model,_,_=load_model(); model=model.cpu(); state=None; pred=[]
    x=torch.from_numpy(np.stack(waves))
    with torch.inference_mode():
        for chunk in x.split(32,-1):
            y,state=model.stream_step(chunk,state); pred.append(y)
        raw=torch.cat(pred,1)
        torch.testing.assert_close(raw,model(x)[0],rtol=2e-4,atol=2e-5)
    p=clamp(raw.numpy())
    out=HERE/'results'/('two_station_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir()
    np.save(out/'station_predictions.npy',p)
    rows=[]; summaries=[]
    plt.rcParams.update({'font.family':['SimSun','DejaVu Sans'],'axes.unicode_minus':False})
    for i,event in enumerate(events):
        first,second=event['selected'][:2]; headers=[r['header'] for r in (first,second)]
        stations=[(h['station_latitude'],h['station_longitude']) for h in headers]
        true=(headers[0]['latitude'],headers[0]['longitude'])
        delay=(datetime.fromisoformat(second['trigger_time'])-datetime.fromisoformat(first['trigger_time'])).total_seconds()
        active=None; local=[]
        for j in range(93):
            t=(j+1)*.32+offsets[2*i]
            k=math.floor((t-delay-offsets[2*i+1]+1e-9)/.32)-1
            a=p[2*i,j]; az1=float(np.degrees(np.arctan2(a[2],a[3]))%360)
            single=location(*stations[0],a[1],az1)
            mag=float(a[0]); joint=single; status='first_only'; gap=None
            # At least 4 seconds of new-station input: first eligible block is 4.16s.
            ready=0<=k<93 and (k+1)*.32>=4.
            if ready:
                b=p[2*i+1,k]; az2=float(np.degrees(np.arctan2(b[2],b[3]))%360)
                mag=float((a[0]+b[0])/2)
                joint,status,gap=solve(stations,[float(a[1]),float(b[1])],[az1,az2])
                if active is None: active=t
            row=dict(event=event['name'],date=event['date'],first_record=first['record'],second_record=second['record'],
                time_since_first_p_s=t,second_trigger_delay_s=delay,second_chunk=k+1 if k>=0 else None,
                first_magnitude=float(a[0]),joint_magnitude=mag,true_magnitude=headers[0]['magnitude'],
                first_location_error_km=distance(single,true),joint_location_error_km=distance(joint,true) if joint is not None else None,
                predicted_latitude=float(joint[0]) if joint is not None else None,predicted_longitude=float(joint[1]) if joint is not None else None,
                geometry=status,candidate_score_gap_deg=gap)
            rows.append(row);local.append(row)
        d=pd.DataFrame(local)
        fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True,layout='constrained')
        t=d.time_since_first_p_s
        axes[0].plot(t,d.first_magnitude,label='首台独立',color='#888888')
        axes[0].plot(t,d.joint_magnitude,label='双台联合（接入前沿用首台）',color='#2166ac')
        axes[0].axhline(headers[0]['magnitude'],color='#d6604d',ls='--',label='真实震级')
        axes[0].set_ylabel('震级')
        axes[1].plot(t,d.first_location_error_km,label='首台独立',color='#888888')
        axes[1].plot(t,d.joint_location_error_km,label='双台联合（含不相交回退）',color='#2166ac')
        axes[1].set(ylabel='震中定位误差 / km',xlabel='首台P到时后的时间 / s')
        for ax in axes:
            ax.axvline(delay,color='#999999',ls=':',label='第二台P到时')
            if active is not None: ax.axvline(active,color='#2a9d8f',ls='--',label='双台开始联合')
            ax.grid(alpha=.2);ax.legend(fontsize=9)
        fig.suptitle(f"{event['date']} {event['name']}｜{first['station']} + {second['station']}")
        fig.savefig(out/f"{event['date']}.png",dpi=180);fig.savefig(out/f"{event['date']}.pdf");plt.close(fig)
        summaries.append(dict(event=event['name'],second_delay_s=delay,fusion_start_s=active,
            geometry_counts=d.geometry.value_counts().to_dict(),final=local[-1]))
    pd.DataFrame(rows).to_csv(out/'replay.csv',index=False,encoding='utf-8-sig')
    (out/'results.json').write_text(json.dumps(dict(source_h5=str(SOURCE),weights=str(WEIGHTS),selection=str(selection),
        protocol='CPU FP32, original 32-point streams; absolute time alignment with H5 P offsets; second input >=4s (4.16s grid); no future samples; equal mean magnitude; spherical circle intersections selected by summed circular azimuth errors; no truth used in selection; distance-only least-squares fallback',
        scope='six descriptive events; annotated trigger times, no detection/telemetry delays; event-wise errors, no calibrated uncertainty',events=summaries),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(out,flush=True)


if __name__=='__main__': main()
