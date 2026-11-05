"""Plot saved K-NET triplets and annotated P arrivals, one figure per record."""
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULT = HERE/'results/20261022_230231_957410/first_stations.json'


def read_wave(path):
    lines=Path(path).read_text(encoding='ascii').splitlines()
    memo=next(i for i,line in enumerate(lines) if line.strip()=='Memo.')
    header=lines[:memo]
    fs=float(re.search(r'([\d.]+)Hz',next(s for s in header if s.startswith('Sampling Freq'))).group(1))
    scale=next(s for s in header if s.startswith('Scale Factor'))
    match=re.search(r'([\d.eE+-]+)\(gal\)/([\d.eE+-]+)',scale)
    if not match: raise ValueError(f'Invalid calibration: {path}')
    num,den=map(float,match.groups())
    wave=np.array([int(v) for line in lines[memo+1:] for v in line.split()],dtype=np.float64)*num/den
    if not len(wave) or not np.isfinite(wave).all(): raise ValueError(path)
    return np.arange(len(wave))/fs,wave,fs


def main():
    report=json.loads(RESULT.read_text(encoding='utf-8'))
    out=RESULT.parent/'waveform_plots'
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.unicode_minus':False})
    plots=[]
    for event in report['events']:
        for row in event['selected']:
            folder=out/f"station_{row['rank']}"
            folder.mkdir(parents=True,exist_ok=True)
            target=folder/f"{event['date']}_{row['record']}.png"
            fig,axes=plt.subplots(3,1,figsize=(12,6),sharex=True,layout='constrained')
            try:
                for ax,ext,label in zip(axes,['.UD','.NS','.EW'],['Z (UD)','N (NS)','E (EW)']):
                    time,wave,fs=read_wave(row['saved_files'][ext]['path'])
                    p=row['p_arrival_s']
                    if not 0<=p<=time[-1]: raise ValueError(f'P outside waveform: {row["record"]}')
                    ax.plot(time,wave,color='#24384c',linewidth=.55)
                    ax.axvline(p,color='#d62728',linestyle='--',linewidth=1.2)
                    ax.set_ylabel(f'{label}\nGal')
                    ax.grid(alpha=.2)
                    ax.set_xlim(0,time[-1])
                axes[0].annotate(f'P = {p:g} s',xy=(p,1),xycoords=('data','axes fraction'),
                                 xytext=(5,-5),textcoords='offset points',va='top',color='#d62728',
                                 bbox=dict(facecolor='white',edgecolor='none',alpha=.8))
                axes[-1].set_xlabel('Time since record start (s)')
                fig.suptitle(f"{event['date']} | {row['station']} | Station rank {row['rank']}\n"
                             f"Record start: {row['header']['record_time']} | P arrival: {row['trigger_time']}",fontsize=11)
                fig.savefig(target,dpi=180)
                plots.append(dict(event=event['date'],record=row['record'],rank=row['rank'],
                                  p_arrival_s=p,path=str(target)))
            finally: plt.close(fig)
    (out/'manifest.json').write_text(json.dumps(dict(processing='fixed scale to gal only; no filtering, detrending or normalization',
         source=str(RESULT),plots=plots),ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Saved {len(plots)} plots: {out}')


if __name__=='__main__': main()
