"""First-station raw triplet above four saved prediction trajectories."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plot_first_waveforms import read_wave

HERE=Path(__file__).resolve().parent
RUN=HERE/'results/compactgru_first_20261022_231155_186077'


def main():
    events=json.loads((HERE/'results/20261022_230231_957410/first_stations.json').read_text(encoding='utf-8'))['events']
    data=pd.read_csv(RUN/'trajectory_plots/trajectory_data.csv')
    out=RUN/'combined_first_station';out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':['SimSun','DejaVu Sans'],'axes.unicode_minus':False,'font.size':10})
    for event in events:
        row=event['selected'][0]
        d=data[data.record==row['record']].sort_values('nominal_post_p_s')
        fig=plt.figure(figsize=(13,12),layout='constrained')
        grid=fig.add_gridspec(5,2,height_ratios=[.8,.8,.8,1.4,1.4])
        for i,(ext,label) in enumerate(zip(['.UD','.NS','.EW'],['Z / gal','N / gal','E / gal'])):
            ax=fig.add_subplot(grid[i,:])
            t,w,fs=read_wave(row['saved_files'][ext]['path'])
            t=t-row['p_arrival_s']
            mask=(t>=-5)&(t<=30)
            ax.plot(t[mask],w[mask],color='#24384c',lw=.6)
            ax.axvline(0,color='#d62728',ls='--',lw=1.2,label='标注P到时')
            ax.set(xlim=(-5,30),ylabel=label)
            ax.grid(alpha=.2)
            if i==0:ax.legend(loc='upper right')
            if i==2:ax.set_xlabel('相对标注P到时 / s')
        specs=[('predicted_magnitude','true_magnitude','震级估计','震级'),
               ('predicted_distance_km','true_distance_km','震中距估计','距离 / km'),
               ('predicted_azimuth_deg','true_azimuth_deg','方位角估计','方位角 / °'),
               ('location_error_km',None,'震中定位误差','误差 / km')]
        for i,(pred,truth,title,ylabel) in enumerate(specs):
            ax=fig.add_subplot(grid[3+i//2,i%2])
            x=d.exact_post_p_s.to_numpy();y=d[pred].to_numpy()
            if pred=='predicted_azimuth_deg':
                xx=[];yy=[]
                for j,(t,v) in enumerate(zip(x,y)):
                    if j and abs(v-y[j-1])>180:xx.append(np.nan);yy.append(np.nan)
                    xx.append(t);yy.append(v)
                ax.plot(xx,yy,color='#2166ac',lw=1.4,label='预测值')
                ax.set_ylim(0,360)
            else:ax.plot(x,y,color='#2166ac',lw=1.4,label='预测值' if truth else '定位误差')
            if truth:ax.axhline(d[truth].iloc[0],color='#d6604d',ls='--',label=f'真实值 {d[truth].iloc[0]:.2f}')
            ax.set(xlim=(-5,30),xlabel='相对标注P到时 / s',ylabel=ylabel,title=title)
            ax.axvline(0,color='#d62728',ls='--',lw=.8)
            ax.grid(alpha=.2);ax.legend(fontsize=9)
        fig.suptitle(f"{event['date']} {event['name']}｜第一台 {row['station']}\n三分量原始波形与 CompactGRU 连续预测",fontsize=14)
        for suffix in ('png','pdf'):fig.savefig(out/f"{event['date']}_{row['station']}.{suffix}",dpi=180)
        plt.close(fig)
    (out/'README.txt').write_text('六次事件第一台站。上方：原始完整记录截取P前5秒至P后30秒，仅固定仪器系数换算gal，不滤波；无数据部分不补零。下方：既有因果H5推理输出，使用exact_post_p_s含裁剪对齐偏移，未平滑。四图为震级、震中距、方位角和震中定位误差。',encoding='utf-8')
    print(out)


if __name__=='__main__':main()
