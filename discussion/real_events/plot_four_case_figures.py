"""Four publication plates; original P annotations; >=3s new-station input."""
from pathlib import Path
from datetime import datetime
import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
HERE=ROOT/'discussion/real_events'
sys.path.insert(0,str(HERE))
import plot_els_six_single_station as paper
import plot_els_event_cases as shared
from land_background import add_land_sea
OUT=ROOT/'plotting/Fig11_Separated_Event_Cases'


def replay():
    run=HERE/'results'/('wait3s_original_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));run.mkdir()
    original=HERE/'results/20261022_230231_957410/first_stations.json'
    (run/'first_stations.json').write_text(original.read_text(encoding='utf-8'),encoding='utf-8')
    for name in ('two_station_replay.py','three_station_replay.py'):
        text=(HERE/name).read_text(encoding='utf-8')
        text=text.replace('Path(__file__).resolve().parent','Path('+repr(str(HERE))+')')
        text=text.replace('results/20261022_230231_957410/first_stations.json',str(run/'first_stations.json').replace('\\','/'))
        text=text.replace("HERE/'results'/",'Path('+repr(str(run))+')/')
        text=text.replace('>=4.','>=3.').replace('>=4s','>=3s').replace('4.16s','3.20s')
        path=run/name;path.write_text(text,encoding='utf-8')
        ns={'__name__':'case_replay','__file__':str(path)};exec(compile(text,str(path),'exec'),ns);ns['main']()
    return run


def new_figure(height_mm=250):
    fig,axes=plt.subplots(3,2,figsize=(190/25.4,height_mm/25.4))
    return fig,axes.flat


def finish(fig,name,handles):
    fig.tight_layout(rect=(0,.065,1,1),h_pad=2,w_pad=2)
    fig.legend(handles=handles,loc='lower center',ncol=min(3,len(handles)),frameon=False,fontsize=8)
    for ext in ('pdf','png'):fig.savefig(OUT/(name+'.'+ext),dpi=600)
    fig.savefig(OUT/(name+'_preview.png'),dpi=130)
    plt.close(fig)


def map_plate(cases,frames,name,stage_colors):
    fig,axes=new_figure()
    for i,(ax,c,d) in enumerate(zip(axes,cases,frames)):
        h=c['row']['header'];x=d.longitude.to_numpy();y=d.latitude.to_numpy();st=d.stage.to_numpy()
        for j in range(1,len(x)):
            ax.plot(x[j-1:j+1],y[j-1:j+1],color=stage_colors[st[j]],lw=1.5)
        for stage,color in stage_colors.items():
            mask=st==stage;ax.scatter(x[mask],y[mask],s=5,color=color)
        ax.scatter(x[0],y[0],s=16,facecolors='white',edgecolors=stage_colors[st[0]],zorder=6)
        ax.scatter(x[-1],y[-1],s=16,color=stage_colors[st[-1]],zorder=6)
        ax.scatter(h['station_longitude'],h['station_latitude'],marker='v',s=30,color='black',zorder=7)
        # Show all station positions used by that plate.
        for r in c['event']['selected'][1:len(stage_colors)]:
            hh=r['header'];ax.scatter(hh['station_longitude'],hh['station_latitude'],marker='v',s=25,color='black',zorder=7)
        ax.scatter(h['longitude'],h['latitude'],marker='*',s=85,color='#D62728',edgecolor='black',lw=.3,zorder=7)
        ax.set(xlabel='Longitude (°E)',ylabel='Latitude (°N)',title=f'({chr(97+i)}) {shared.TITLES[i]}')
        ax.title.set_fontsize(10);ax.xaxis.set_major_locator(MaxNLocator(3));ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.ticklabel_format(style='plain',useOffset=False)
    fig.tight_layout(rect=(0,.065,1,1),h_pad=2,w_pad=2)
    fig.canvas.draw()
    for ax,c,d in zip(fig.axes,cases,frames):
        hh=[r['header'] for r in c['event']['selected'][:len(stage_colors)]]
        h=hh[0]
        xx=np.r_[d.longitude,[v['station_longitude'] for v in hh],h['longitude']]
        yy=np.r_[d.latitude,[v['station_latitude'] for v in hh],h['latitude']]
        paper.geographic_limits(ax,xx,yy,h['latitude'])
        add_land_sea(ax)
    handles=[Line2D([],[],color=color,lw=1.5,marker='o',markersize=3,label=stage.title()+' station') for stage,color in stage_colors.items()]
    handles += [Line2D([],[],marker='v',ls='',color='black',label='Station'),Line2D([],[],marker='*',ls='',color='#D62728',markersize=9,label='Target Quake')]
    fig.legend(handles=handles,loc='lower center',ncol=3,frameon=False,fontsize=8)
    for ext in ('pdf','png'):fig.savefig(OUT/(name+'.'+ext),dpi=600)
    fig.savefig(OUT/(name+'_preview.png'),dpi=130);plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    shared.USE_CORRECTED_NOTO=False
    cases=shared.load_cases();paper.set_paper_style()
    run=replay()
    two=pd.read_csv(next(run.glob('two_station_*/replay.csv')))
    three=pd.read_csv(next(run.glob('three_station_*/replay.csv')))
    fig,axes=new_figure(height_mm=140)
    for i,(ax,c) in enumerate(zip(axes,cases)):
        paper.plot_wave_mag(ax,c);ax.set_title(f'({chr(97+i)}) {shared.TITLES[i]} — {c["row"]["station"]}',fontsize=10,loc='left')
    handles=[Line2D([],[],color='#555555',lw=.5,label='Z waveform')]
    handles += [Line2D([],[],color=color,lw=1.5,label=label) for color,label in zip(shared.COLORS,['0–10 s','10–20 s','20–29.76 s'])]
    handles += [Line2D([],[],color='#D62728',ls='--',label='True magnitude / P arrival')]
    finish(fig,'01_Waveform_Magnitude',handles)
    single=[];dual=[];triple=[]
    for c in cases:
        date=int(c['event']['date']);a=two[two.date==date].reset_index(drop=True);b=three[three.date==date].reset_index(drop=True)
        # Existing single predictions use WGS84 display coordinates; fused methods use spherical solver.
        single.append(pd.DataFrame(dict(longitude=c['lon'],latitude=c['lat'],stage='single')))
        d=pd.DataFrame(dict(longitude=a.predicted_longitude,latitude=a.predicted_latitude,stage=np.where(a.geometry=='first_only','single','two')))
        dual.append(d.copy())
        mask=b.distance_azimuth_latitude.notna()
        d.loc[mask,'longitude']=b.loc[mask,'distance_azimuth_longitude']
        d.loc[mask,'latitude']=b.loc[mask,'distance_azimuth_latitude'];d.loc[mask,'stage']='three';triple.append(d)
    map_plate(cases,single,'02_Single_Station_Tracks',{'single':'#D4A017'})
    map_plate(cases,dual,'03_Two_Station_Wait3s_Tracks',{'single':'#D4A017','two':'#0072B2'})
    map_plate(cases,triple,'04_Three_Station_Wait3s_Tracks',{'single':'#D4A017','two':'#0072B2','three':'#009E73'})
    (OUT/'manifest.json').write_text(json.dumps(dict(replay=str(run),picks='Original spreadsheet; Noto ISK00313.52s, no correction',
        station_selection='Original chronological first three from original selection; Chiba original order CHB027/CHB020/CHB018',
        wait_seconds=3,first_new_station_output_seconds=3.2,three_method='distance + azimuth robust fit, exploratory10km/30deg scales',
        geometry='single WGS84 display; fused positions original spherical solver; no predictions altered',
        figures=4,format=['vector PDF','600 DPI PNG']),indent=2),encoding='utf-8')
    import shutil;shutil.copy2(__file__,OUT/Path(__file__).name)
    print('FOUR FIGURES:',OUT,flush=True)


if __name__=='__main__':main()
