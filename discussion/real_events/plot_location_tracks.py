"""Longitude/latitude tracks: first station -> two stations -> three stations."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE=Path(__file__).resolve().parent


def main():
    revision=Path(json.loads((HERE/'results/latest_revision.json').read_text())['directory'])
    events=json.loads((revision/'first_stations.json').read_text(encoding='utf-8'))['events']
    two=pd.read_csv(sorted(revision.glob('two_station_*/replay.csv'))[-1])
    three=pd.read_csv(sorted(revision.glob('three_station_*/replay.csv'))[-1])
    out=revision/'location_tracks';out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':['SimSun','DejaVu Sans'],'axes.unicode_minus':False,'font.size':10})
    all_rows=[]
    for event in events:
        a=two[two.date==int(event['date'])].reset_index(drop=True)
        b=three[three.date==int(event['date'])].reset_index(drop=True)
        np.testing.assert_allclose(a.time_since_first_p_s,b.time_since_first_p_s)
        d=a[['date','time_since_first_p_s','predicted_latitude','predicted_longitude']].copy()
        d['stage']=np.where(a.geometry=='first_only','single','two')
        mask=b.distance_azimuth_latitude.notna()
        d.loc[mask,'stage']='three'
        d.loc[mask,'predicted_latitude']=b.loc[mask,'distance_azimuth_latitude']
        d.loc[mask,'predicted_longitude']=b.loc[mask,'distance_azimuth_longitude']
        all_rows.append(d)
        fig,ax=plt.subplots(figsize=(10,8),layout='constrained')
        colors={'single':'#dfae00','two':'#2166ac','three':'#239b56'}
        names={'single':'单台定位','two':'双台定位','three':'三台距离＋方位定位'}
        x=d.predicted_longitude.to_numpy();y=d.predicted_latitude.to_numpy()
        stages=d.stage.to_numpy()
        # Each segment uses the newly available stage; preserve transition jumps.
        for j in range(1,len(d)):
            ax.plot(x[j-1:j+1],y[j-1:j+1],color=colors[stages[j]],lw=1.4,alpha=.85,zorder=2)
        for stage,color in colors.items():
            m=stages==stage
            if m.any():
                ax.scatter(x[m],y[m],s=14,color=color,zorder=3)
                idx=np.flatnonzero(m)[0]
                ax.annotate(f"{names[stage]}开始\n{d.time_since_first_p_s.iloc[idx]:.2f}s",
                    (x[idx],y[idx]),xytext=(8,10),textcoords='offset points',fontsize=8,
                    bbox=dict(facecolor='white',alpha=.7,edgecolor='none'))
        h=event['selected'][0]['header']
        ax.scatter(h['longitude'],h['latitude'],marker='*',s=320,color='#d62728',edgecolor='black',lw=.6,zorder=6)
        ax.annotate('真实震中',(h['longitude'],h['latitude']),xytext=(8,-15),textcoords='offset points',color='#b22222')
        for n,r in enumerate(event['selected'][:3],1):
            v=r['header'];ax.scatter(v['station_longitude'],v['station_latitude'],marker='^',s=100,
                color='#333333',edgecolor='white',lw=.6,zorder=5)
            ax.annotate(f"{n} {r['station']}",(v['station_longitude'],v['station_latitude']),
                        xytext=(7,5 if n!=2 else -15),textcoords='offset points',fontsize=9)
        ax.scatter(x[-1],y[-1],s=75,facecolors='none',edgecolors=colors[stages[-1]],lw=1.5,zorder=5)
        ax.annotate('最后估计',(x[-1],y[-1]),xytext=(7,-13),textcoords='offset points',fontsize=8)
        handles=[Line2D([],[],color=c,marker='o',markersize=4,label=names[s]+('（本事件无独立阶段）' if not (stages==s).any() else '')) for s,c in colors.items()]
        handles += [Line2D([],[],marker='^',ls='',color='#333333',label='台站'),Line2D([],[],marker='*',ls='',color='#d62728',markersize=12,label='真实震中')]
        ax.legend(handles=handles,fontsize=9,loc='best')
        ax.set(xlabel='经度 / °E',ylabel='纬度 / °N',title=event['date']+' '+event['name']+'｜依次接入三台的定位轨迹')
        ax.set_aspect(1/np.cos(np.radians(h['latitude'])))
        ax.grid(alpha=.25);ax.ticklabel_format(useOffset=False,style='plain')
        fig.savefig(out/(event['date']+'.png'),dpi=200);fig.savefig(out/(event['date']+'.pdf'));plt.close(fig)
    pd.concat(all_rows).to_csv(out/'track_coordinates.csv',index=False,encoding='utf-8-sig')
    (out/'manifest.json').write_text(json.dumps(dict(revision=str(revision),three_station_method='distance + azimuth, exploratory scales 10km/30deg',
        transition='actual fusion availability after >=4s input, not P trigger instant',
        lines='all saved estimates connected without smoothing; transition segment uses new stage color',
        warning='A blue-only stage may be absent when two/three fusion becomes available on the same output grid point'),indent=2),encoding='utf-8')
    print(out)


if __name__=='__main__':main()
