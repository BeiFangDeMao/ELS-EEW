"""Six four-panel prediction trajectories from saved CompactGRU inference."""
import json
from pathlib import Path
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
RUN=HERE/'results/compactgru_first_20261022_231155_186077'
SELECTION=HERE/'results/20261022_230231_957410/first_stations.json'


def location(lat,lon,distance,azimuth):
    lat,lon=np.radians([lat,lon]); angle=np.radians(azimuth); delta=distance/6371.
    la=np.arcsin(np.clip(np.sin(lat)*np.cos(delta)+np.cos(lat)*np.sin(delta)*np.cos(angle),-1,1))
    lo=lon+np.arctan2(np.sin(angle)*np.sin(delta)*np.cos(lat),np.cos(delta)-np.sin(lat)*np.sin(la))
    return np.degrees(la),(np.degrees(lo)+180)%360-180


def main():
    data=pd.read_csv(RUN/'predictions_by_time.csv')
    events=json.loads(SELECTION.read_text(encoding='utf-8'))['events']
    out=RUN/'trajectory_plots'; out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':['SimSun','DejaVu Sans'],'axes.unicode_minus':False,'font.size':10})
    tables=[]
    for event in events:
        row=event['selected'][0]; h=row['header']
        d=data[data.record==row['record']].sort_values('nominal_post_p_s').copy()
        if len(d)!=93: raise ValueError(row['record'])
        lat,lon=location(h['station_latitude'],h['station_longitude'],
                         d.predicted_distance_km.to_numpy(),d.predicted_azimuth_deg.to_numpy())
        a=np.sin((np.radians(lat)-math.radians(h['latitude']))/2)**2
        a+=np.cos(np.radians(lat))*math.cos(math.radians(h['latitude']))*np.sin((np.radians(lon)-math.radians(h['longitude']))/2)**2
        error=12742*np.arcsin(np.sqrt(np.clip(a,0,1)))
        d['predicted_latitude']=lat; d['predicted_longitude']=lon; d['location_error_km']=error
        tables.append(d)
        fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
        specs=[('predicted_magnitude','true_magnitude','震级估计','震级'),
               ('predicted_distance_km','true_distance_km','震中距估计','距离 / km'),
               ('predicted_azimuth_deg','true_azimuth_deg','方位角估计','方位角 / °'),
               ('location_error_km',None,'震中定位误差','误差 / km')]
        for ax,(pred,true,title,ylabel) in zip(axes.flat,specs):
            x=d.nominal_post_p_s.to_numpy(); y=d[pred].to_numpy()
            if pred=='predicted_azimuth_deg':
                # Do not draw misleading straight lines across the 0/360 boundary.
                xx=[]; yy=[]
                for i,(t,v) in enumerate(zip(x,y)):
                    if i and abs(v-y[i-1])>180: xx.append(np.nan); yy.append(np.nan)
                    xx.append(t); yy.append(v)
                ax.plot(xx,yy,color='#2166ac',linewidth=1.5,label='预测值')
                ax.set_ylim(0,360); ax.set_yticks(np.arange(0,361,90))
            else: ax.plot(x,y,color='#2166ac',linewidth=1.5,label='定位误差' if true is None else '预测值')
            if true: ax.axhline(d[true].iloc[0],color='#d6604d',linestyle='--',label=f'真实值 {d[true].iloc[0]:.2f}')
            else: ax.axhline(0,color='#555555',linewidth=.7)
            ax.set(title=title,xlabel='P后观测时间 / s',ylabel=ylabel,xlim=(0,30))
            ax.grid(alpha=.25); ax.legend(fontsize=9)
        fig.suptitle(f"{event['date']} {event['name']}｜首台 {row['station']}｜CompactGRU",fontsize=14)
        fig.savefig(out/f"{event['date']}_{row['station']}.png",dpi=200)
        fig.savefig(out/f"{event['date']}_{row['station']}.pdf")
        plt.close(fig)
    pd.concat(tables).to_csv(out/'trajectory_data.csv',index=False,encoding='utf-8-sig')
    (out/'manifest.json').write_text(json.dumps(dict(source=str(RUN),figures=6,
        time='original nominal P-aligned grid 0.32–29.76 s; per-record alignment offsets retained in CSV',
        localization='station + predicted epicentral distance and azimuth; great-circle error to archive header epicenter; radius 6371 km',
        curves='93 saved outputs; no smoothing; azimuth wrap crossings disconnected'),indent=2),encoding='utf-8')
    print(out)


if __name__=='__main__': main()
