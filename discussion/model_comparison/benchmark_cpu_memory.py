"""Isolated-process CPU memory measurement on the latency benchmark waveform."""
import argparse,csv,gc,json,platform,subprocess,sys,threading,time
from datetime import datetime
from pathlib import Path
import h5py
import numpy as np
import psutil
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

def child(name):
    # Import both implementations for every worker before measuring the runtime baseline.
    sys.path.insert(0,str(HERE))
    from train_comparison import make_model
    from comparison_dataset import ComparisonDataset
    sys.path.insert(0,str(ROOT/'els_eew'))
    from models.compact_gru import CompactGRU
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    run=HERE/'training_results/20261022_212803_440932'
    identity=json.loads((run/'identity.json').read_text())
    # Register upstream imports in all processes, independent of the selected model.
    import train_comparison
    train_comparison.upstream()
    data=ComparisonDataset(Path(identity['dataset']),'test','emg')
    wave,_,key=data[0];data.close()
    x=torch.from_numpy(wave[None]).contiguous();chunks=tuple(x.split(32,dim=-1))
    process=psutil.Process()
    def snap():
        m=process.memory_info()
        return dict(rss=m.rss,private=m.private,os_peak_rss=m.peak_wset)
    gc.collect();baseline=snap()
    if name=='ELS-EEW':
        model=CompactGRU().eval()
        weights=ROOT/'els_eew/outputs/experiment_20261012_223631_490021/CompactGRU_seed42/checkpoint_CompactGRU_20261012_235806/best_CompactGRU_20261012_235806.pth'
        ckpt=torch.load(weights,map_location='cpu',weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'],strict=True);del ckpt
        models=[model]
    else:
        names=['seist_s_emg','seist_s_dis','seist_s_baz'] if name=='SeisT-S' else ['magnet']
        models=[]
        for model_name in names:
            model=make_model(model_name,identity['config']).eval()
            ckpt=torch.load(run/model_name/'best.pth',map_location='cpu',weights_only=False)
            model.load_state_dict(ckpt['model'],strict=True);del ckpt
            models.append(model)
    gc.collect();loaded=snap()
    peaks=dict(loaded);stop=threading.Event();started=threading.Event();samples=[0]
    def sample():
        started.set()
        while not stop.is_set():
            m=snap();samples[0]+=1
            for k,v in m.items():peaks[k]=max(peaks[k],v)
            stop.wait(.001)
    sampler=threading.Thread(target=sample,daemon=True);sampler.start();started.wait()
    state=None
    with torch.inference_mode():
        for iteration in range(100):
            if name=='ELS-EEW':
                state=None
                for chunk in chunks:output,state=models[0].stream_step(chunk,state)
            else:output=tuple(m(x) for m in models)
            if iteration==0:first=snap()
        final=snap()
    stop.set();sampler.join()
    for snapshot in [first,final]:
        for k,v in snapshot.items():peaks[k]=max(peaks[k],v)
    def tensor_bytes(value):
        if torch.is_tensor(value):return value.numel()*value.element_size()
        if isinstance(value,(tuple,list)):return sum(tensor_bytes(v) for v in value)
        return 0
    tensors=sum(p.numel()*p.element_size() for m in models for p in m.parameters())
    tensors+=sum(p.numel()*p.element_size() for m in models for p in m.buffers())
    mib=1024**2
    result=dict(model=name,record=key,baseline_rss_mib=baseline['rss']/mib,
        loaded_rss_mib=loaded['rss']/mib,inference_peak_rss_mib=peaks['rss']/mib,
        peak_rss_increment_over_baseline_mib=(peaks['rss']-baseline['rss'])/mib,
        peak_rss_increment_over_loaded_mib=(peaks['rss']-loaded['rss'])/mib,
        inference_peak_private_mib=peaks['private']/mib,
        os_lifetime_peak_rss_mib=peaks['os_peak_rss']/mib,
        parameter_and_buffer_kib=tensors/1024,persistent_stream_state_kib=tensor_bytes(state)/1024,
        rss_samples=samples[0],replays=100)
    print(json.dumps(result))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--child');args=parser.parse_args()
    if args.child:child(args.child);return
    result=HERE/'cpu_memory_results'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    result.mkdir(parents=True)
    rows=[]
    for repeat in range(3):
        for name in ['ELS-EEW','SeisT-S','MagNet']:
            output=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--child',name],capture_output=True,text=True,check=True)
            row=json.loads(output.stdout.strip());row['repeat']=repeat+1;rows.append(row)
            print('Completed',name,'repeat',repeat+1,flush=True)
    summary=[]
    for name in ['ELS-EEW','SeisT-S','MagNet']:
        selected=[r for r in rows if r['model']==name]
        summary.append(dict(model=name,**{k:float(np.median([r[k] for r in selected]))
            for k in selected[0] if k not in ['model','record','repeat']}))
    for filename,values in [('memory_individual_runs.csv',rows),('memory_summary.csv',summary)]:
        with (result/filename).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    report=dict(cpu=platform.processor(),torch=torch.__version__,device='cpu',threads=1,batch=1,
        duration_s=2.88,independent_processes_per_model=3,aggregation='Median of three independent fresh processes',
        rss='Resident working set, MiB=1024^2 bytes; includes Python, PyTorch and native runtime',
        measurement='psutil memory_info sampled every 1 ms during 100 replays, including cold first inference; sampled RSS peak may miss sub-ms transients',
        os_peak='OS lifetime peak includes imports and checkpoint loading; NOT an isolated inference-only peak',
        baseline='Same framework/model imports and real input resident, before model instantiation/loading',
        seist='Three separate single-task models simultaneously resident, executed sequentially',
        els='Nine incremental updates, states reset for each record replay, 32 new samples per update',
        tensors='Exact logical parameter+buffer bytes; excludes allocator overhead; state=retained recurrent/conv tensor bytes',
        rows=summary,runs=rows)
    (result/'memory_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (result.parent/'latest_run.json').write_text(json.dumps(dict(directory=str(result)),indent=2))
    print(json.dumps(dict(directory=str(result),summary=summary),indent=2),flush=True)

if __name__=='__main__':main()
