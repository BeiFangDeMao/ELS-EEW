"""Publication figure: six real single-station cases; no inference/training.

Noto explicitly uses the user-corrected ISK003 P pick and matching predictions.
Run with lightweight Python. GeographicLib is installed locally, not in training env.
"""
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import numpy as np
import pandas as pd
from PIL import Image

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
HERE=ROOT/'discussion/real_events'
sys.path.insert(0,str(HERE/'plot_dependencies'))
from geographiclib.geodesic import Geodesic
from plot_first_waveforms import read_wave

BASE=HERE/'results/20261022_230231_957410'
PRED=HERE/'results/compactgru_first_20261022_231155_186077'
CORRECTED=HERE/'results/corrected_noto_first_20261022_233748_413996'
OUT=ROOT/'plotting/Fig10_Real_Event_Cases'
COLORS=['#D4A017','#0072B2','#009E73']
TITLES=['2024 Noto Peninsula','2024 Hyuganada','2024 Bungo Channel',
        '2023 Noto','2024 Fukushima Offshore','2023 Southern Chiba']
STATIONS=['ISK003','MYZ013','EHM012','ISK002','FKS011','CHB027']
USE_CORRECTED_NOTO=True


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_cases():
    events=json.loads((BASE/'first_stations.json').read_text(encoding='utf-8'))['events']
    correction=json.loads((CORRECTED/'manifest.json').read_text(encoding='utf-8'))
    predictions=np.load(PRED/'processed_predictions.npy',allow_pickle=False)
    truths=np.load(PRED/'truths.npy',allow_pickle=False)
    assert predictions.shape==(6,93,4)
    cases=[]
    for i,event in enumerate(events):
        row=event['selected'][0]
        assert row['station']==STATIONS[i]
        corrected=i==0 and USE_CORRECTED_NOTO
        pick=correction['corrected_p_seconds'] if corrected else row['p_arrival_s']
        p=np.load(CORRECTED/'predictions.npy',allow_pickle=False) if corrected else predictions[i]
        assert p.shape==(93,4) and np.isfinite(p).all()
        channels=[]
        for ext in ('.UD','.NS','.EW'):
            source=row['saved_files'][ext]
            assert sha(source['path'])==source['sha256']
            time,wave,rate=read_wave(source['path'])
            if pick<3 or time[-1]<pick+29.76:
                raise ValueError(f'Real pre/post-P coverage missing: {source["path"]}')
            t=time-pick
            mask=(t>=-3)&(t<=29.76)
            # Display-only baseline removal, retaining gal and common gain.
            noise=(t>=-3)&(t<0)
            baseline=float(np.mean(wave[noise]))
            channels.append(dict(t=t[mask],wave=wave[mask]-baseline,baseline_gal=baseline,
                                 path=source['path'],sha256=source['sha256']))
        if corrected:
            assert correction['record']==row['record'] and abs(pick-19.02)<1e-9
            saved=pd.read_csv(CORRECTED/'predictions.csv')
            np.testing.assert_allclose(saved.magnitude,p[:,0],rtol=1e-6)
        azi=np.degrees(np.arctan2(p[:,2],p[:,3]))%360
        h=row['header']
        coordinates=[Geodesic.WGS84.Direct(h['station_latitude'],h['station_longitude'],
                      float(a),float(d)*1000) for d,a in zip(p[:,1],azi)]
        lon=np.array([v['lon2'] for v in coordinates]);lat=np.array([v['lat2'] for v in coordinates])
        for d,a,la,lo in zip(p[:,1],azi,lat,lon):
            check=Geodesic.WGS84.Inverse(h['station_latitude'],h['station_longitude'],la,lo)
            assert abs(check['s12']/1000-float(d))<1e-6
        cases.append(dict(event=event,row=row,pick=pick,pred=p,truth=truths[i],channels=channels,
            time=np.arange(1,94)*.32,lon=lon,lat=lat,
            prediction_source=str(CORRECTED/'predictions.npy' if corrected else PRED/'processed_predictions.npy')))
    return cases


def stage(t):
    return 0 if t<=10 else 1 if t<=20 else 2


def colored_curve(ax,x,y,time):
    # No interpolation: every original segment retained, colored by ending time.
    points=np.column_stack([x,y])
    segments=np.stack([points[:-1],points[1:]],axis=1)
    ax.add_collection(LineCollection(segments,colors=[COLORS[stage(t)] for t in time[1:]],
                                    linewidths=1.5,zorder=4))


def amplitude_scale(value):
    exponent=10**np.floor(np.log10(max(value,1e-10)))
    return max(v*exponent for v in (0.1,0.2,0.5,1,2,5) if v*exponent<=value)


def draw_wave_magnitude(ax,case):
    peak=max(np.max(np.abs(c['wave'])) for c in case['channels'])
    offsets=np.array([2.5,0,-2.5])*peak
    for label,c,offset in zip('ZNE',case['channels'],offsets):
        ax.plot(c['t'],c['wave']+offset,color='black',lw=.35,zorder=1)
        ax.text(-2.8,offset+peak*.25,label,fontsize=8,va='bottom',
                bbox=dict(facecolor='white',edgecolor='none',pad=.3,alpha=.8))
    ax.set(xlim=(-3,29.76),ylim=(-4*peak,4*peak),xlabel='Time after P arrival (s)',
           ylabel='Acceleration (gal; offset)')
    ax.set_yticks([])
    ax.set_xticks([0,10,20,29.76]);ax.set_xticklabels(['0','10','20','29.76'])
    ax.axvline(0,color='#D62728',ls='--',lw=.65,zorder=3)
    scale=amplitude_scale(peak)
    x=-1.6;y=-3.6*peak
    ax.plot([x,x],[y,y+scale],color='black',lw=.8)
    ax.text(x+.7,y+scale/2,f'{scale:g} gal',fontsize=7,va='center')
    twin=ax.twinx()
    colored_curve(twin,case['time'],case['pred'][:,0],case['time'])
    truth=float(case['truth'][0]);twin.axhline(truth,color='#D62728',ls='--',lw=.8)
    low=min(case['pred'][:,0].min(),truth);high=max(case['pred'][:,0].max(),truth)
    pad=max(.15,(high-low)*.12)
    twin.set(ylim=(low-pad,high+pad),ylabel='Magnitude')
    twin.yaxis.set_major_locator(MaxNLocator(4))
    return twin


def draw_track(ax,case):
    h=case['row']['header'];x=case['lon'];y=case['lat']
    colored_curve(ax,x,y,case['time'])
    ax.scatter(x[0],y[0],s=16,facecolors='white',edgecolors=COLORS[0],lw=.8,zorder=6)
    ax.scatter(x[-1],y[-1],s=14,color=COLORS[-1],zorder=6)
    ax.scatter(h['longitude'],h['latitude'],s=95,marker='*',color='#D62728',edgecolor='black',lw=.35,zorder=7)
    ax.scatter(h['station_longitude'],h['station_latitude'],s=35,marker='^',color='black',zorder=7)
    all_x=np.r_[x,h['longitude'],h['station_longitude']]
    all_y=np.r_[y,h['latitude'],h['station_latitude']]
    dx=max(np.ptp(all_x),.04);dy=max(np.ptp(all_y),.04)
    ax.set(xlim=(all_x.min()-.12*dx,all_x.max()+.12*dx),
           ylim=(all_y.min()-.12*dy,all_y.max()+.12*dy),xlabel='Longitude (°E)',ylabel='Latitude (°N)')
    # Expand limits to the physical axes rectangle; retain local geographic scale.
    box=ax.get_position();fig=ax.figure
    ratio=(box.height*fig.get_figheight())/(box.width*fig.get_figwidth())
    cosine=np.cos(np.radians(np.mean(all_y)))
    xr=ax.get_xlim();yr=ax.get_ylim()
    span_x=max(xr[1]-xr[0],(yr[1]-yr[0])/(cosine*ratio))
    span_y=span_x*cosine*ratio
    cx=np.mean(xr);cy=np.mean(yr)
    ax.set_xlim(cx-span_x/2,cx+span_x/2);ax.set_ylim(cy-span_y/2,cy+span_y/2)
    ax.xaxis.set_major_locator(MaxNLocator(3));ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))


def draw_figure(cases):
    plt.rcParams.update({'font.family':['Times New Roman','SimSun'],'font.size':9,
        'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,
        'axes.linewidth':.8,'xtick.major.width':.8,'ytick.major.width':.8,
        'pdf.fonttype':42,'ps.fonttype':42,'axes.unicode_minus':False,'savefig.facecolor':'white'})
    fig=plt.figure(figsize=(190/25.4,385/25.4),facecolor='white')
    outer=fig.add_gridspec(3,2,left=.095,right=.905,bottom=.12,top=.975,wspace=.60,hspace=.29)
    for i,case in enumerate(cases):
        inner=outer[i//2,i%2].subgridspec(2,1,height_ratios=[1,1],hspace=.38)
        upper=fig.add_subplot(inner[0]);draw_wave_magnitude(upper,case)
        upper.set_title(f'({chr(97+i)}) {TITLES[i]} — {STATIONS[i]}',fontsize=10,loc='left',pad=10)
        lower=fig.add_subplot(inner[1]);draw_track(lower,case)
    handles=[Line2D([],[],color='black',lw=.5)]
    labels=['Waveform (gal; baseline removed, vertically offset)']
    for color,label in zip(COLORS,['0–10 s','10–20 s','20–29.76 s']):
        handles.append(Line2D([],[],color=color,lw=1.5));labels.append('Predicted magnitude / Epicenter trajectory: '+label)
    handles.append((Line2D([],[],color='#D62728',ls='--',lw=.8),Line2D([],[],color='#D62728',marker='*',ls='',markersize=9)))
    labels.append('True magnitude / Epicenter')
    handles.append(Line2D([],[],color='black',marker='^',ls='',markersize=6));labels.append('Station')
    fig.legend(handles,labels,handler_map={tuple:HandlerTuple(ndivide=None)},loc='lower center',
               bbox_to_anchor=(.5,.006),ncol=1,frameon=False,handlelength=3,labelspacing=.35)
    return fig


def export(cases,fig):
    OUT.mkdir(parents=True,exist_ok=True)
    metadata=[]
    for case in cases:
        stem=case['event']['date']+'_'+case['row']['station']
        pd.DataFrame(dict(time_s=case['time'],magnitude=case['pred'][:,0],distance_km=case['pred'][:,1],
            azimuth_deg=np.degrees(np.arctan2(case['pred'][:,2],case['pred'][:,3]))%360,
            longitude=case['lon'],latitude=case['lat'])).to_csv(OUT/(stem+'_predictions.csv'),index=False)
        metadata.append(dict(record=case['row']['record'],p_seconds=case['pick'],outputs=93,
             source=case['prediction_source'],prediction_sha256=sha(case['prediction_source']),
             waveform_files=[dict(path=c['path'],sha256=c['sha256'],display_baseline_gal=c['baseline_gal']) for c in case['channels']]))
    pdf=OUT/'ELS_EEW_Event_Cases.pdf';png=OUT/'ELS_EEW_Event_Cases.png'
    fig.savefig(pdf);fig.savefig(png,dpi=600)
    fig.savefig(OUT/'preview.png',dpi=140)
    # Check renderer extents for labels escaping figure borders.
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    outside=[]
    for ax in fig.axes:
        box=ax.get_tightbbox(renderer)
        if box.x0<0 or box.y0<0 or box.x1>fig.bbox.width or box.y1>fig.bbox.height:
            outside.append(str(ax.get_position()))
    if outside:raise RuntimeError('Labels outside figure: '+str(outside))
    with Image.open(png) as im:resolution=im.size
    manifest=dict(cases=metadata,width_mm=190,height_mm=385,png_pixels=resolution,dpi=600,
        geodesy='GeographicLib WGS84 direct; all distances inverse-checked; station-to-epicenter azimuth',
        time='Original 93 nominal P-aligned outputs 0.32..29.76s; retained original sample alignment offsets <=0.009s; waveforms relative to annotated P',
        noto='ISK003 original P13.52s replaced by user correction19.02s; matching reprocessed and inferred predictions used; not independently repicked',
        colors='Original segments colored by end-time bin; no prediction smoothing or extra interpolated outputs',
        waveform='Actual -3..29.76s source samples, pre-P mean removed for display only, same gal scale for three offset channels per event',
        note='WGS84 display geometry differs slightly from historical spherical evaluation; predictions unchanged',
        files={pdf.name:sha(pdf),png.name:sha(png)},script_sha256=sha(__file__))
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    import shutil
    if Path(__file__).resolve()!= (OUT/Path(__file__).name).resolve():shutil.copy2(__file__,OUT/Path(__file__).name)
    print(OUT, resolution,flush=True)


def main():
    cases=load_cases();fig=draw_figure(cases);export(cases,fig);plt.close(fig)


if __name__=='__main__':main()
