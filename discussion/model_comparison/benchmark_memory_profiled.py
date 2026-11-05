"""Independent CPU RSS and chronological profiler tensor-memory measurements."""
import argparse,csv,gc,gzip,hashlib,itertools,json,platform,subprocess,sys,threading,time
from datetime import datetime
from pathlib import Path
import numpy as np
import psutil
import h5py
import torch
from torch.profiler import profile,ProfilerActivity

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

def write_csv(path,rows):
    if not rows:return
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def storages(value):
    if torch.is_tensor(value):
        s=value.untyped_storage()
        return {(str(value.device),s.data_ptr()):s.nbytes()} if s.nbytes() else {}
    out={}
    for v in value if isinstance(value,(tuple,list)) else []:out.update(storages(v))
    return out

def resident_bytes(value):return sum(storages(value).values())

def measure_profiler(fn,path,known_resident):
    # Profiler is deliberately separate from RSS measurements.
    with torch.inference_mode(),profile(activities=[ProfilerActivity.CPU],profile_memory=True,
                                       record_shapes=True,with_stack=True) as prof:
        output=fn()
    prof.export_memory_timeline(str(path/'memory_events.raw.json.gz'),device='cpu')
    with gzip.open(path/'memory_events.raw.json.gz','rt') as f:events=json.load(f)
    assert events,'No CPU memory events were captured'
    initial=sum(int(e[2]) for e in events if e[1]==1)
    active=initial;peak=initial;timeline=[]
    # Group equal timestamps; version changes are reclassification, not allocation.
    actual=sorted([e for e in events if e[1]!=1],key=lambda e:e[0])
    for stamp,group in itertools.groupby(actual,key=lambda e:e[0]):
        group=list(group)
        active+=sum(int(e[2]) for e in group)
        assert active>=0,f'Unbalanced memory timeline: {active}'
        peak=max(peak,active)
        timeline.append(dict(timestamp_ns=stamp,active_tracked_bytes=active,
                             delta_bytes=sum(int(e[2]) for e in group)))
    write_csv(path/'active_tensor_timeline.csv',timeline)
    missing=max(0,known_resident-initial)
    report=dict(preexisting_tracked_bytes=initial,known_resident_storage_bytes=known_resident,
        peak_active_tracked_bytes=peak,peak_increment_over_tracked_start_bytes=peak-initial,
        accounted_peak_active_bytes=peak+missing,untracked_resident_bytes_added=missing,
        final_active_tracked_bytes=active,memory_events=len(events),
        definition='Chronological live allocation balance including PREEXISTING; CREATE adds, DESTROY subtracts; version changes net zero. Fixed logical resident storage absent from trace is added once, never summed across operators.',
        scope='Tracked CPU tensor/storage allocations and profiler-visible allocations. Excludes Python objects, DLLs, allocator reserved-but-unused pages and untracked native workspaces. Not total RSS.')
    (path/'profiler_summary.json').write_text(json.dumps(report,indent=2))
    del prof,output;gc.collect()
    return report

def child(name,out):
    out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(HERE))
    from train_comparison import make_model,upstream
    from comparison_dataset import ComparisonDataset
    sys.path.insert(0,str(ROOT/'els_eew'))
    from models.compact_gru import CompactGRU
    upstream() # same implementation imports in every worker
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    run=HERE/'training_results/20261022_212803_440932'
    identity=json.loads((run/'identity.json').read_text())
    data=ComparisonDataset(Path(identity['dataset']),'test','emg')
    wave,_,key=data[0];data.close()
    x=torch.from_numpy(wave[None]).contiguous();chunks=tuple(x.split(32,dim=-1))
    process=psutil.Process()
    snap=lambda:process.memory_info().rss
    # Initialize lightweight process monitor APIs before the baseline.
    snap();gc.collect();baseline=snap()
    models=[];hashes={}
    if name=='ELS-EEW':
        model=CompactGRU().eval()
        weights=ROOT/'els_eew/outputs/experiment_20261012_223631_490021/CompactGRU_seed42/checkpoint_CompactGRU_20261012_235806/best_CompactGRU_20261012_235806.pth'
        ckpt=torch.load(weights,map_location='cpu',weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'],strict=True);del ckpt
        models=[model];hashes[str(weights)]=hashlib.sha256(weights.read_bytes()).hexdigest()
    else:
        for model_name in (['seist_s_emg','seist_s_dis','seist_s_baz'] if name=='SeisT-S' else ['magnet']):
            weights=run/model_name/'best.pth';model=make_model(model_name,identity['config']).eval()
            ckpt=torch.load(weights,map_location='cpu',weights_only=False)
            model.load_state_dict(ckpt['model'],strict=True);del ckpt
            models.append(model);hashes[str(weights)]=hashlib.sha256(weights.read_bytes()).hexdigest()
    gc.collect();loaded=snap()
    def replay():
        if name=='ELS-EEW':
            state=None
            for chunk in chunks:prediction,state=models[0].stream_step(chunk,state)
            return prediction,state
        # All three SeisT weights remain resident; outputs retained through all tasks.
        return tuple(m(x) for m in models)
    with torch.inference_mode():
        for _ in range(100):output=replay();del output
    gc.collect();warmed=snap()
    rss_samples=[];stop=threading.Event()
    def monitor():
        while not stop.is_set():
            rss_samples.append((time.perf_counter_ns(),snap()))
            stop.wait(.001)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    with torch.inference_mode():
        for _ in range(200):output=replay();del output
    stop.set();thread.join()
    rss_samples.append((time.perf_counter_ns(),snap()))
    peak=max(v for _,v in rss_samples)
    write_csv(out/'rss_samples.csv',[dict(timestamp_ns=t,rss_bytes=v) for t,v in rss_samples])
    persistent=0;long_report=None
    if name=='ELS-EEW':
        meta=json.loads((ROOT/'discussion/early_reliability_warning/results/20261018_180137_947801_record_only/inference_manifest.json').read_text())
        with h5py.File(meta['source_h5'],'r') as f:full=torch.from_numpy(np.ascontiguousarray(f[key][:].astype(np.float32).T)[None])
        np.testing.assert_allclose(full[:,:,:288].numpy(),x.numpy(),atol=1e-6,rtol=1e-6)
        full_chunks=tuple(full.split(32,dim=-1));assert len(full_chunks)==93
        state=None;steps=[]
        with torch.inference_mode():
            for index,chunk in enumerate(full_chunks):
                output,state=models[0].stream_step(chunk,state)
                steps.append(dict(chunk=index+1,state_storage_bytes=resident_bytes(state),rss_bytes=snap()))
            persistent=resident_bytes(state)
            assert persistent==2112 and all(r['state_storage_bytes']==persistent for r in steps)
            del output,state
            # Continuous-state stress replay, only for memory stability, not accuracy.
            state=None;blocks=[]
            for block in range(40):
                for chunk in full_chunks:output,state=models[0].stream_step(chunk,state)
                blocks.append(dict(block=block+1,updates=(block+1)*93,state_storage_bytes=resident_bytes(state),rss_bytes=snap()))
            assert all(r['state_storage_bytes']==persistent for r in blocks)
            del output,state
        write_csv(out/'stream_93_updates.csv',steps);write_csv(out/'continuous_stream_stability.csv',blocks)
        steady=np.array([r['rss_bytes'] for r in blocks[10:]],dtype=float)
        long_report=dict(updates_per_record=93,continuous_stress_updates=3720,state_bytes=persistent,
            state_constant_all_93_and_3720=True,steady_rss_range_mib=float(np.ptp(steady)/2**20),
            steady_first10_median_rss_mib=float(np.median(steady[:10])/2**20),
            steady_last10_median_rss_mib=float(np.median(steady[-10:])/2**20),
            note='Real waveform repeated with state retained as a memory-only stress test. First 10 blocks excluded only from plateau summary, all raw measurements saved. Not a prediction-quality experiment.')
        del full,full_chunks;gc.collect()
    weights_bytes=resident_bytes([list(m.parameters())+list(m.buffers()) for m in models])
    profile_dir=out/'profile_288';profile_dir.mkdir()
    prof=measure_profiler(replay,profile_dir,weights_bytes+resident_bytes(x))
    long_prof=None
    if name=='ELS-EEW':
        # Fresh full-record storage for a separately scoped 93-update trace.
        with h5py.File(meta['source_h5'],'r') as f:full=torch.from_numpy(np.ascontiguousarray(f[key][:].astype(np.float32).T)[None])
        full_chunks=tuple(full.split(32,dim=-1))
        def full_replay():
            state=None
            for chunk in full_chunks:prediction,state=models[0].stream_step(chunk,state)
            return prediction,state
        p=out/'profile_2976';p.mkdir()
        long_prof=measure_profiler(full_replay,p,weights_bytes+resident_bytes(full))
    result=dict(model=name,record=key,baseline_rss_mib=baseline/2**20,loaded_rss_mib=loaded/2**20,
        warmed_rss_mib=warmed/2**20,warm_inference_peak_rss_mib=peak/2**20,
        load_delta_rss_mib=(loaded-baseline)/2**20,peak_over_baseline_mib=(peak-baseline)/2**20,
        peak_over_loaded_mib=(peak-loaded)/2**20,peak_over_warm_mib=(peak-warmed)/2**20,
        rss_sample_count=len(rss_samples),median_actual_sample_interval_ms=float(np.median(np.diff([t for t,_ in rss_samples]))/1e6),
        weight_buffer_kib=weights_bytes/1024,persistent_state_kib=persistent/1024,
        profiler_accounted_peak_kib=prof['accounted_peak_active_bytes']/1024,
        profiler_increment_over_start_kib=prof['peak_increment_over_tracked_start_bytes']/1024,
        profiler=prof,long_stream=long_report,long_profiler=long_prof,weights_sha256=hashes)
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--child');parser.add_argument('--output',type=Path);args=parser.parse_args()
    if args.child:child(args.child,args.output);return
    folder=HERE/'cpu_profiled_memory_results'/datetime.now().strftime('%Y%m%d_%H%M%S_%f');folder.mkdir(parents=True)
    results=[]
    for repeat in range(3):
        for name in ['ELS-EEW','SeisT-S','MagNet']:
            path=folder/f'{name}_repeat{repeat+1}'
            p=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--child',name,'--output',str(path)],capture_output=True,text=True)
            (folder/f'{name}_repeat{repeat+1}.log').write_text(p.stderr,encoding='utf-8')
            if p.returncode:raise RuntimeError(p.stderr)
            result=json.loads(p.stdout.strip());results.append(result)
            print('Completed',name,'independent process',repeat+1,flush=True)
    rows=[]
    for name in ['ELS-EEW','SeisT-S','MagNet']:
        selected=[r for r in results if r['model']==name]
        rows.append(dict(model=name,**{k:float(np.median([r[k] for r in selected])) for k,v in selected[0].items() if isinstance(v,(int,float))}))
    write_csv(folder/'memory_summary.csv',rows)
    report=dict(cpu=platform.processor(),pytorch=torch.__version__,device='cpu',fp32=True,batch=1,
        threads=1,interop_threads=1,independent_processes=3,warmup=100,rss_measured_replays=200,
        summary_aggregation='Median across three fresh processes; retain per-run JSON, RSS samples, profiler events and active-byte timelines',
        profiler_excluded_from_rss_measurement=True,rss_sample_interval_requested_ms=1,
        baseline='Same imports and same test input prepared, before model construction',
        tensor_scope='Weights+buffers+entire input storage+profiler-visible active temporaries and outputs/states. Unused resident weights absent from profiler preexisting records added once. No double-counted operator allocations.',
        limitation='RSS sampled, actual interval recorded. Profiler-visible active tensor memory excludes untracked native workspaces/allocator reserve/Python framework; not total RAM. Native allocator caches can stabilize at different RSS values.',
        rows=rows,runs=results)
    (folder/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (folder.parent/'latest_run.json').write_text(json.dumps(dict(directory=str(folder)),indent=2))
    print(json.dumps(dict(directory=str(folder),summary=rows),indent=2),flush=True)

if __name__=='__main__':main()
