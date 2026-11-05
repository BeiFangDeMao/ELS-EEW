"""Compare equal spherical mean of per-station epicenters against two-circle solver."""
from pathlib import Path
import sys
import json
import math
from datetime import datetime
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import two_station_replay as geo
import plot_four_case_figures as plots
import plot_els_event_cases as shared


def main():
    source=HERE/'results/wait3s_original_20261023_003844_729241'
    prior=next(source.glob('two_station_*'))
    pred=np.load(prior/'station_predictions.npy',allow_pickle=False)
    baseline=pd.read_csv(prior/'replay.csv')
    metadata=json.loads((prior/'results.json').read_text(encoding='utf-8'))
    report=json.loads((source/'first_stations.json').read_text(encoding='utf-8'))
    out=HERE/'results'/('direct_two_station_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir()
    all_rows=[];summaries=[];frames=[]
    for i,event in enumerate(report['events']):
        records=event['selected'][:2]
        stations=[(r['header']['station_latitude'],r['header']['station_longitude']) for r in records]
        truth=(records[0]['header']['latitude'],records[0]['header']['longitude'])
        old=baseline[baseline.date==int(event['date'])].reset_index(drop=True)
        rows=[]
        for j,r in old.iterrows():
            a=pred[2*i,j]
            azi=float(np.degrees(np.arctan2(a[2],a[3]))%360)
            first=geo.location(*stations[0],float(a[1]),azi)
            position=first;stage='single';second=None
            if r.geometry!='first_only':
                k=int(r.second_chunk)-1
                b=pred[2*i+1,k]
                az2=float(np.degrees(np.arctan2(b[2],b[3]))%360)
                second=geo.location(*stations[1],float(b[1]),az2)
                summed=geo.vector(first)+geo.vector(second)
                if np.linalg.norm(summed)<1e-10:raise ValueError('Antipodal position ambiguity')
                position=geo.point(summed);stage='two'
            error=geo.distance(position,truth)
            row=dict(date=event['date'],event=event['name'],time_since_first_p_s=float(r.time_since_first_p_s),
                stage=stage,longitude=float(position[1]),latitude=float(position[0]),
                first_longitude=float(first[1]),first_latitude=float(first[0]),
                second_longitude=float(second[1]) if second is not None else None,
                second_latitude=float(second[0]) if second is not None else None,
                direct_error_km=error,old_circle_error_km=float(r.joint_location_error_km),
                first_error_km=float(r.first_location_error_km),magnitude=float(r.joint_magnitude))
            rows.append(row)
        d=pd.DataFrame(rows);all_rows.append(d);frames.append(d)
        fused=d[d.stage=='two']
        summaries.append(dict(event=event['name'],stations=[r['station'] for r in records],
            fusion_start_s=float(fused.time_since_first_p_s.iloc[0]),
            final_direct_error_km=float(d.direct_error_km.iloc[-1]),
            final_old_circle_error_km=float(d.old_circle_error_km.iloc[-1]),
            final_first_error_km=float(d.first_error_km.iloc[-1]),
            fused_time_mean_direct_error_km=float(fused.direct_error_km.mean()),
            fused_time_mean_old_error_km=float(fused.old_circle_error_km.mean())))
    pd.concat(all_rows).to_csv(out/'replay.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(summaries).to_csv(out/'comparison.csv',index=False,encoding='utf-8-sig')
    (out/'results.json').write_text(json.dumps(dict(method='Each station distance+azimuth -> spherical epicenter; normalize sum of two unit vectors (equal spherical midpoint)',
        source_predictions=str(prior),waiting='new station input >=3s, first eligible output3.20s; original timestamps, no future data',
        calibration='equal weights fixed a priori; no ground truth used in fusion; six descriptive cases only',
        events=summaries),ensure_ascii=False,indent=2),encoding='utf-8')
    shared.USE_CORRECTED_NOTO=False;cases=shared.load_cases();plots.paper.set_paper_style();plots.OUT=out
    plots.map_plate(cases,frames,'Direct_Two_Station_Tracks',{'single':'#D4A017','two':'#0072B2'})
    fig,axes=plots.new_figure()
    for ax,event,d in zip(axes,report['events'],all_rows):
        t=d.time_since_first_p_s
        ax.plot(t,d.first_error_km,color='#999999',lw=1,label='First station')
        ax.plot(t,d.old_circle_error_km,color='#d6604d',lw=1.2,label='Circle + azimuth')
        ax.plot(t,d.direct_error_km,color='#0072B2',lw=1.5,label='Direct position mean')
        fused=d[d.stage=='two'];ax.axvline(fused.time_since_first_p_s.iloc[0],ls=':',color='#555555',lw=.8)
        ax.set(title=shared.TITLES[len([a for a in fig.axes if a.get_title()])-1] if False else event['date'],xlabel='Time after first P (s)',ylabel='Location error / km')
    handles=[plots.Line2D([],[],color=c,lw=1.5,label=l) for c,l in [('#999999','First station'),('#d6604d','Circle + azimuth'),('#0072B2','Direct position mean')]]
    plots.finish(fig,'Location_Error_Comparison',handles)
    print(out,flush=True)
    for e in summaries:print(e['event'],round(e['final_old_circle_error_km'],2),round(e['final_direct_error_km'],2),flush=True)


if __name__=='__main__':main()
