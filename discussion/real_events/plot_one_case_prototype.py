"""One rectangular single-station case prototype; original annotated P pick."""
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
HERE=ROOT/'discussion/real_events'
sys.path.insert(0,str(HERE))
import plot_els_event_cases as shared
import plot_els_six_single_station as paper
from land_background import add_land_sea


def main():
    shared.USE_CORRECTED_NOTO=False
    cases=shared.load_cases();c=cases[0]
    common_peak=max(np.max(np.abs(v['channels'][0]['wave']+v['channels'][0]['baseline_gal'])) for v in cases)
    common_limit=np.ceil(common_peak/100)*100
    cmap=plt.get_cmap('viridis');norm=Normalize(0,29.76)
    def gradient(ax,x,y,t,linewidth=1.5):
        points=np.column_stack([x,y]);segments=np.stack([points[:-1],points[1:]],axis=1)
        lines=LineCollection(segments,cmap=cmap,norm=norm,linewidth=linewidth,zorder=4)
        lines.set_array((t[:-1]+t[1:])/2);ax.add_collection(lines)
        return lines
    paper.set_paper_style()
    out=ROOT/'plotting/Fig12_Single_Event_Prototype';out.mkdir(parents=True,exist_ok=True)
    # Equal plot widths: square map, 16:9 waveform; no gap between frames.
    figure_width_mm=115
    plot_width_mm=figure_width_mm*(.82-.18)
    figure_height_mm=plot_width_mm*(1+9/16)/(.90-.10)
    fig=plt.figure(figsize=(figure_width_mm/25.4,figure_height_mm/25.4))
    gs=fig.add_gridspec(2,1,height_ratios=[16,9],hspace=0,
                        left=.18,right=.82,top=.90,bottom=.10)
    top=fig.add_subplot(gs[0]);h=c['row']['header']
    gradient(top,c['lon'],c['lat'],c['time'],linewidth=.8)
    top.scatter(c['lon'],c['lat'],s=5,c=c['time'],cmap=cmap,norm=norm,zorder=4)
    top.scatter(h['station_longitude'],h['station_latitude'],marker='^',s=55,color='black',zorder=7)
    top.scatter(h['longitude'],h['latitude'],marker='*',s=130,color='#D62728',edgecolors='black',lw=.4,zorder=7)
    top.xaxis.tick_top();top.tick_params(axis='x',labeltop=True,labelbottom=False)
    paper.geographic_limits(top,np.r_[c['lon'],h['station_longitude'],h['longitude']],
                             np.r_[c['lat'],h['station_latitude'],h['latitude']],h['latitude'],pad=.22)
    add_land_sea(top)
    top.ticklabel_format(style='plain',useOffset=False)
    top.xaxis.set_major_locator(paper.MaxNLocator(3));top.yaxis.set_major_locator(paper.MaxNLocator(3))
    top.xaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{x:.1f}°E'))
    top.yaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{x:.1f}°N'))
    # Only real saved outputs: never interpolate to manufacture a 3s prediction.
    for index,offset,label in [(8,(10,-18),'3 s (output 2.88 s)'),
                               (30,(-20,30),'9.92 s'),
                               (61,(12,15),'19.84 s'),
                               (92,(12,-22),'29.76 s')]:
        x=c['lon'][index];y=c['lat'][index]
        top.scatter(x,y,s=28,facecolor='white',edgecolor=cmap(norm(c['time'][index])),lw=1,zorder=8)
        top.annotate(label,(x,y),xytext=offset,textcoords='offset points',fontsize=8,
                     arrowprops=dict(arrowstyle='-',color='#444444',lw=.6),
                     bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=1),zorder=9)
    top.set_title('2024 Noto Peninsula — ISK003',fontsize=10,pad=28)
    lower=fig.add_subplot(gs[1])
    channel=c['channels'][0]
    lower.plot(channel['t'],channel['wave']+channel['baseline_gal'],color='#555555',lw=.4)
    lower.axvline(0,color='#D62728',ls='--',lw=.7)
    lower.set(xlim=(-3,29.76),ylim=(-common_limit,common_limit),ylabel='Waveform / gal',xlabel='Time after P arrival (s)')
    lower.set_yticks([-common_limit,0,common_limit])
    lower.set_xticks([0,10,20,29.76]);lower.set_xticklabels(['0','10','20','29.76'])
    twin=lower.twinx();gradient(twin,c['time'],c['pred'][:,0],c['time'])
    twin.axhline(c['truth'][0],color='#D62728',ls='--',lw=.8)
    twin.set(ylim=(3,8),ylabel='Magnitude');twin.set_yticks([4,6,8])
    lower.axvline(2.88,color='#777777',ls=':',lw=.8)
    handles=[Line2D([],[],marker='^',ls='',color='black',label='Station'),
             Line2D([],[],marker='*',ls='',color='#D62728',markersize=9,label='Epicenter')]
    handles.append(Line2D([],[],color='#D62728',ls='--',label='True magnitude / P'))
    top.legend(handles=handles,loc='upper left',fontsize=8,frameon=True,facecolor='white',edgecolor='none')
    color_axis=top.inset_axes([.54,.90,.40,.025])
    cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=color_axis,orientation='horizontal')
    cb.set_ticks([0,10,20,29.76]);cb.set_ticklabels(['0','10','20','29.76'])
    cb.ax.tick_params(labelsize=7,length=2);cb.set_label('Time after P (s)',fontsize=8,labelpad=1)
    for ext in ('pdf','png'):fig.savefig(out/('Noto_single_station_prototype.'+ext),dpi=600)
    fig.savefig(out/'preview.png',dpi=160)
    import json
    (out/'display_settings.json').write_text(json.dumps(dict(common_six_event_waveform_ylim_gal=[-common_limit,common_limit],
        waveform='raw calibrated Z; no normalization',gradient='viridis; shared time normalization0..29.76s',
        original_p_seconds=c['pick'],map_width_height_ratio=1,
        waveform_width_height_ratio=16/9),indent=2),encoding='utf-8')
    plt.close(fig)
    print(out)


if __name__=='__main__':main()
