"""Exclude Noto ISK003, regenerate plots and sequential 1/2/3-station replay."""
import copy
from datetime import datetime
import json
from pathlib import Path
import sys
import shutil

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent


def main():
    sys.path.insert(0,str(HERE))
    old=HERE/'results/20261022_230231_957410/first_stations.json'
    rev=HERE/'results'/('restored_original_wait4s_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    rev.mkdir()
    report=json.loads(old.read_text(encoding='utf-8'))
    from select_first_stations import rank_event
    event=next(e for e in report['events'] if e['date']=='20240101')
    picks={r['record']:r['p_arrival_s'] for r in event['ranked_records']}
    event.update(rank_event(Path(event['archive']),picks,rev))
    event['explicitly_excluded']=[]
    report['revision']=dict(source=str(old),action='Restore original Noto ISK003, ISK007, TYM001 with original unshifted P picks; wait4s')
    selection=rev/'first_stations.json'
    selection.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    # Execute copies with explicit new paths; original scripts/results remain intact.
    sources=rev/'scripts';sources.mkdir()
    first_run=None
    def execute(name):
        nonlocal first_run
        text=(HERE/name).read_text(encoding='utf-8')
        text=text.replace("Path(__file__).resolve().parent","Path("+repr(str(HERE))+")")
        # Keep HERE a Path after replacing its assignment expression.
        text=text.replace('HERE='+repr(str(HERE)), 'HERE=Path('+repr(str(HERE))+')')
        text=text.replace('results/20261022_230231_957410/first_stations.json',str(selection).replace('\\','/'))
        if first_run is not None:
            text=text.replace('results/compactgru_first_20261022_231155_186077',str(first_run).replace('\\','/'))
        text=text.replace("HERE/'results'/",repr(str(rev)).join([]) if False else "Path("+repr(str(rev))+")/")
        script=sources/name;script.write_text(text,encoding='utf-8')
        namespace={'__name__':'revision_module','__file__':str(script)}
        exec(compile(text,str(script),'exec'),namespace)
        namespace['main']()
        if name=='infer_first_compactgru.py':
            first_run=sorted(rev.glob('compactgru_first_*'))[-1]
    for name in ['plot_first_waveforms.py','infer_first_compactgru.py','plot_first_predictions.py',
                 'plot_first_combined.py','two_station_replay.py','three_station_replay.py']:
        execute(name)
    two=sorted(rev.glob('two_station_*'))[-1]
    three=sorted(rev.glob('three_station_*'))[-1]
    a=pd.read_csv(two/'replay.csv');b=pd.read_csv(three/'replay.csv')
    out=rev/'sequential_localization';out.mkdir()
    plt.rcParams.update({'font.family':['SimSun','DejaVu Sans'],'axes.unicode_minus':False})
    tables=[]
    for e in report['events']:
        d=a[a.date==int(e['date'])].copy();q=b[b.date==int(e['date'])].copy()
        d=d.reset_index(drop=True);q=q.reset_index(drop=True)
        if len(d)!=len(q):raise ValueError('time grid mismatch')
        active=q.distance_only_error_km.notna()
        d['stage']=d.geometry.map(lambda v:'single' if v=='first_only' else 'two')
        d.loc[active,'stage']='three'
        d['sequential_distance_error_km']=d.joint_location_error_km
        d['sequential_combined_error_km']=d.joint_location_error_km
        d.loc[active,'sequential_distance_error_km']=q.loc[active,'distance_only_error_km']
        d.loc[active,'sequential_combined_error_km']=q.loc[active,'distance_azimuth_error_km']
        d['sequential_magnitude']=d.joint_magnitude
        d.loc[active,'sequential_magnitude']=q.loc[active,'three_magnitude']
        tables.append(d)
        fig,axes=plt.subplots(2,1,figsize=(12,8),sharex=True,layout='constrained')
        t=d.time_since_first_p_s
        axes[0].plot(t,d.first_magnitude,color='#999999',label='首台独立')
        axes[0].plot(t,d.sequential_magnitude,color='#2166ac',label='依次接入台站的等权平均震级')
        axes[0].axhline(d.true_magnitude.iloc[0],color='#d6604d',ls='--',label='真实震级')
        axes[0].set_ylabel('震级')
        axes[1].plot(t,d.first_location_error_km,color='#999999',label='首台独立')
        axes[1].plot(t,d.sequential_distance_error_km,color='#2166ac',label='单台→双台→三台仅距离')
        axes[1].plot(t,d.sequential_combined_error_km,color='#e69f00',label='单台→双台→三台距离＋方位')
        axes[1].set(ylabel='震中定位误差 / km',xlabel='首台标注P到时后 / s')
        origin=datetime.fromisoformat(e['selected'][0]['trigger_time'])
        for n,r in enumerate(e['selected'][1:],2):
            delay=(datetime.fromisoformat(r['trigger_time'])-origin).total_seconds()
            for ax in axes:ax.axvline(delay,color='#aaaaaa',ls=':',label=f'第{n}台P到时 {delay:.2f}s')
        for stage,color,label in [('two','#2a9d8f','双台联合开始'),('three','#8e44ad','三台联合开始')]:
            mask=(d.stage!='single') if stage=='two' else (d.stage=='three')
            if mask.any():
                when=t[mask].iloc[0]
                for ax in axes:ax.axvline(when,color=color,ls='--',label=f'{label} {when:.2f}s')
        for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8,loc='best')
        fig.suptitle(e['date']+' '+e['name']+'｜'+' → '.join(r['station'] for r in e['selected']))
        for ext in ('png','pdf'):fig.savefig(out/(e['date']+'.'+ext),dpi=180)
        plt.close(fig)
    pd.concat(tables).to_csv(out/'sequential_replay.csv',index=False,encoding='utf-8-sig')
    (HERE/'results/latest_revision.json').write_text(json.dumps(dict(directory=str(rev),selection=str(selection)),indent=2),encoding='utf-8')
    print('REVISION COMPLETE:',rev,flush=True)


if __name__=='__main__':main()
