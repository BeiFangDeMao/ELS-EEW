"""Six real single-station cases, map above Z waveform/magnitude.

Run with lightweight Python. Reuses audited sources and local GeographicLib.
No inference, retraining, prediction edits or synthetic waveform padding.
"""
from pathlib import Path
import json
import shutil
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
HERE=ROOT/'discussion/real_events'
sys.path.insert(0,str(HERE))
import plot_els_event_cases as shared
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import numpy as np
import pandas as pd
from PIL import Image

# ---------------- User-editable parameters ----------------
SELECTION_PATH=shared.BASE/'first_stations.json'
PREDICTION_DIR=shared.PRED
CORRECTED_NOTO_DIR=shared.CORRECTED
OUTPUT_DIR=ROOT/'plotting/Fig10_Real_Event_Cases'
EVENT_ORDER=['20240101','20240808','20240417','20230505','20240315','20230511']
WAVEFORM_COMPONENT='Z'  # Z, N or E; consistent across all events
ENABLE_INSET=False  # Optional; off for the final figure to avoid obscuring real tracks
USE_CORRECTED_NOTO=False  # Latest instruction: original spreadsheet P pick (13.52 s)
FIGURE_WIDTH_MM=190
FIGURE_HEIGHT_MM=350
COLORS=shared.COLORS
STEM='ELS_EEW_6events_single_station_cases'


def set_paper_style():
    plt.rcParams.update({'font.family':['Times New Roman','SimSun'],'font.size':9,
        'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,
        'axes.linewidth':.8,'xtick.major.width':.8,'ytick.major.width':.8,
        'axes.unicode_minus':False,'pdf.fonttype':42,'ps.fonttype':42,
        'figure.facecolor':'white','savefig.facecolor':'white'})


def destination_point(lat,lon,distance_km,azimuth_deg):
    result=shared.Geodesic.WGS84.Direct(lat,lon,azimuth_deg,distance_km*1000)
    return result['lat2'],result['lon2']


def split_time_segments(t_pred):
    t=np.asarray(t_pred)
    return [(t>=0)&(t<=10),(t>10)&(t<=20),(t>20)&(t<=29.76+1e-9)]


def geographic_limits(ax,x,y,lat_reference,pad=.15):
    dx=max(np.ptp(x),.02);dy=max(np.ptp(y),.02)
    box=ax.get_position();fig=ax.figure
    ratio=box.height*fig.get_figheight()/(box.width*fig.get_figwidth())
    cosine=np.cos(np.radians(lat_reference))
    dx=max(dx*(1+2*pad),dy*(1+2*pad)/(ratio*cosine));dy=dx*ratio*cosine
    cx=(np.min(x)+np.max(x))/2;cy=(np.min(y)+np.max(y))/2
    ax.set_xlim(cx-dx/2,cx+dx/2);ax.set_ylim(cy-dy/2,cy+dy/2)


def map_symbols(ax,event,small=False):
    h=event['row']['header']
    shared.colored_curve(ax,event['lon'],event['lat'],event['time'])
    for mask,color in zip(split_time_segments(event['time']),COLORS):
        ax.scatter(event['lon'][mask],event['lat'][mask],s=3 if small else 5,color=color,zorder=4)
    ax.scatter(h['station_longitude'],h['station_latitude'],marker='v',s=15 if small else 35,color='black',zorder=7)
    ax.scatter(h['longitude'],h['latitude'],marker='*',s=35 if small else 90,color='#D62728',edgecolor='black',lw=.25,zorder=7)
    ax.scatter(event['lon'][0],event['lat'][0],s=10,facecolor='white',edgecolor=COLORS[0],lw=.7,zorder=8)
    ax.scatter(event['lon'][-1],event['lat'][-1],s=10,color=COLORS[-1],zorder=8)


def plot_event_map(ax,event):
    h=event['row']['header'];map_symbols(ax,event)
    x=np.r_[event['lon'],h['longitude'],h['station_longitude']]
    y=np.r_[event['lat'],h['latitude'],h['station_latitude']]
    geographic_limits(ax,x,y,h['latitude'],pad=.2)
    ax.set(xlabel='Longitude (°E)',ylabel='Latitude (°N)')
    ax.xaxis.set_major_locator(MaxNLocator(3));ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'));ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    if ENABLE_INSET:
        # Attach outside the data rectangle: inset cannot hide any full trajectory points.
        inset=ax.inset_axes([.62,.61,.35,.35])
        inset.set_in_layout(False)
        map_symbols(inset,event,small=True)
        # Robust late-track focus plus actual epicenter; no samples removed from main map.
        focus=event['time']>20
        xx=np.r_[event['lon'][focus],h['longitude']]
        yy=np.r_[event['lat'][focus],h['latitude']]
        geographic_limits(inset,xx,yy,h['latitude'],pad=.15)
        inset.set_xticks([]);inset.set_yticks([])
        inset.set_title('Zoom',fontsize=8,pad=2)


def plot_wave_mag(ax,event):
    c=event['channels'][{'Z':0,'N':1,'E':2}[WAVEFORM_COMPONENT]]
    # Undo the previous figure's display baseline correction: use actual calibrated raw gal.
    raw=c['wave']+c['baseline_gal']
    ax.plot(c['t'],raw,color='#555555',lw=.4,zorder=1)
    ax.axvline(0,color='#D62728',ls='--',lw=.7)
    ax.set(xlim=(-3,29.76),xlabel='Time after P arrival (s)',ylabel='Waveform / gal')
    ax.set_xticks([0,10,20,29.76]);ax.set_xticklabels(['0','10','20','29.76'])
    ax.yaxis.set_major_locator(MaxNLocator(3));ax.ticklabel_format(axis='y',style='plain',useOffset=False)
    twin=ax.twinx()
    shared.colored_curve(twin,event['time'],event['pred'][:,0],event['time'])
    true=event['truth'][0];twin.axhline(true,color='#D62728',ls='--',lw=.8)
    low=min(event['pred'][:,0].min(),true);high=max(event['pred'][:,0].max(),true)
    margin=max(.15,(high-low)*.15)
    twin.set(ylim=(low-margin,high+margin),ylabel='Magnitude')
    twin.yaxis.set_major_locator(MaxNLocator(3))


def plot_event_panel(fig,outer_spec,event,panel_label):
    inner=outer_spec.subgridspec(2,1,height_ratios=[1.65,1])
    ax=fig.add_subplot(inner[0]);plot_event_map(ax,event)
    ax.set_title(f'({panel_label}) {event["title"]} — {event["row"]["station"]}',loc='left',fontsize=10,pad=8)
    bottom=fig.add_subplot(inner[1]);plot_wave_mag(bottom,event)


def main():
    if WAVEFORM_COMPONENT not in ('Z','N','E'):raise ValueError('Unsupported component')
    shared.BASE=SELECTION_PATH.parent;shared.PRED=PREDICTION_DIR;shared.CORRECTED=CORRECTED_NOTO_DIR
    shared.USE_CORRECTED_NOTO=USE_CORRECTED_NOTO
    cases=shared.load_cases()
    by_date={c['event']['date']:c for c in cases}
    titles={c['event']['date']:title for c,title in zip(cases,shared.TITLES)}
    cases=[by_date[d] for d in EVENT_ORDER]
    for c in cases:
        c['title']=titles[c['event']['date']]
        assert len(c['time'])==93 and sum(m.sum() for m in split_time_segments(c['time']))==93
    set_paper_style()
    fig=plt.figure(figsize=(FIGURE_WIDTH_MM/25.4,FIGURE_HEIGHT_MM/25.4))
    grid=fig.add_gridspec(4,2,height_ratios=[1,1,1,.20])
    for i,event in enumerate(cases):plot_event_panel(fig,grid[i//2,i%2],event,chr(97+i))
    legend_ax=fig.add_subplot(grid[3,:]);legend_ax.axis('off')
    handles=[Line2D([],[],marker='v',ls='',color='black',label='Station'),
             Line2D([],[],marker='*',ls='',color='#D62728',markersize=9,label='Target Quake'),
             Line2D([],[],marker='o',lw=1.5,color='#555555',markersize=3,label='Predicted Quake')]
    handles += [Line2D([],[],lw=1.5,color=color,label=label) for color,label in zip(COLORS,['0–10 s','10–20 s','20–29.76 s'])]
    handles.append(Line2D([],[],color='#D62728',ls='--',label='True Magnitude'))
    legend_ax.legend(handles=handles,ncol=3,loc='center',frameon=False,columnspacing=1.5)
    # Constrained layout first fixes panel sizes, then update geographic limits.
    fig.tight_layout(pad=1.3,h_pad=1.8,w_pad=2.2)
    fig.canvas.draw()
    # Geographic ratios computed before constrained layout must be recalculated.
    for ax,event in zip([a for a in fig.axes if a.get_xlabel()=='Longitude (°E)'],cases):
        h=event['row']['header']
        geographic_limits(ax,np.r_[event['lon'],h['longitude'],h['station_longitude']],
                          np.r_[event['lat'],h['latitude'],h['station_latitude']],h['latitude'],pad=.2)
        if ax.child_axes:
            inset=ax.child_axes[0];focus=event['time']>20
            geographic_limits(inset,np.r_[event['lon'][focus],h['longitude']],
                               np.r_[event['lat'][focus],h['latitude']],h['latitude'])
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    for ext in ('pdf','png'):fig.savefig(OUTPUT_DIR/(STEM+'.'+ext),dpi=600)
    fig.savefig(OUTPUT_DIR/(STEM+'_preview.png'),dpi=140)
    for c in cases:
        pd.DataFrame(dict(time_s=c['time'],magnitude=c['pred'][:,0],distance_km=c['pred'][:,1],
            azimuth_deg=np.degrees(np.arctan2(c['pred'][:,2],c['pred'][:,3]))%360,
            longitude=c['lon'],latitude=c['lat'])).to_csv(OUTPUT_DIR/(c['event']['date']+'_single_station_data.csv'),index=False)
    with Image.open(OUTPUT_DIR/(STEM+'.png')) as im:size=im.size
    manifest=dict(component=WAVEFORM_COMPONENT,inset=ENABLE_INSET,png_pixels=size,width_mm=FIGURE_WIDTH_MM,
        height_mm=FIGURE_HEIGHT_MM,geodesy='WGS84; GeographicLib direct/inverse checked',
        noto_p_seconds=cases[0]['pick'],noto_predictions=cases[0]['prediction_source'],
        waveform='raw calibrated gal, no baseline subtraction/filtering/normalization; real pre-P3s',
        time='93 nominal P-aligned outputs .32..29.76; inherited original sample alignment offset <=.009s',
        inset_note='late-track + true epicenter extent; station symbol drawn but may lie outside zoom; main view contains all points',
        files={ext:shared.sha(OUTPUT_DIR/(STEM+'.'+ext)) for ext in ('pdf','png')})
    (OUTPUT_DIR/(STEM+'_manifest.json')).write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    for path in (Path(__file__),Path(shared.__file__)):
        target=OUTPUT_DIR/path.name
        if path.resolve()!=target.resolve():shutil.copy2(path,target)
    plt.close(fig);print(OUTPUT_DIR/STEM,flush=True)


if __name__=='__main__':main()
