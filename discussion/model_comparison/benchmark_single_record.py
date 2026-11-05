"""CPU-only latency on one real shared test waveform; never trains a model."""
import csv
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
import h5py
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
from train_comparison import make_model
from comparison_dataset import ComparisonDataset

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    run=HERE/'training_results/20261022_212803_440932'
    identity=json.loads((run/'identity.json').read_text())
    data=ComparisonDataset(Path(identity['dataset']),'test','emg')
    wave,_,key=data[0]
    data.close()
    x=torch.from_numpy(wave[None]).contiguous()
    project=ROOT/'els_eew'
    sys.path.insert(0,str(project))
    from models.compact_gru import CompactGRU
    weights=project/'outputs/experiment_20261012_223631_490021/CompactGRU_seed42/checkpoint_CompactGRU_20261012_235806/best_CompactGRU_20261012_235806.pth'
    checkpoint=torch.load(weights,map_location='cpu',weights_only=False)
    compact=CompactGRU().eval()
    compact.load_state_dict(checkpoint['model_state_dict'],strict=True)
    manifest=json.loads((ROOT/'discussion/early_reliability_warning/results/20261018_180137_947801_record_only/inference_manifest.json').read_text())
    with h5py.File(manifest['source_h5'],'r') as f:
        old=np.ascontiguousarray(f[key][:288].astype(np.float32).T)
    np.testing.assert_allclose(old,wave,atol=1e-6,rtol=1e-6)
    chunks=tuple(x.split(32,dim=-1))
    def stream():
        state=None
        for part in chunks:out,state=compact.stream_step(part,state)
        return out
    with torch.inference_mode():
        target=compact(x)[0][:,-1:]
        np.testing.assert_allclose(stream().numpy(),target.numpy(),rtol=2e-5,atol=1e-4)
    models={};hashes={str(weights):sha(weights)}
    for name in ['seist_s_emg','seist_s_dis','seist_s_baz','magnet']:
        path=run/name/'best.pth'
        model=make_model(name,identity['config']).eval()
        model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model'],strict=True)
        models[name]=model
        hashes[str(path)]=sha(path)
    def all_seist():
        return tuple(models[name](x) for name in ['seist_s_emg','seist_s_dis','seist_s_baz'])
    funcs={'ELS-EEW_stream_9_updates':stream,'SeisT_three_tasks':all_seist,
           **{name:(lambda name=name:models[name](x)) for name in models}}
    repeats=1000;warmup=100
    samples={name:[] for name in funcs}
    order=list(funcs)
    rng=np.random.default_rng(42)
    with torch.inference_mode():
        for name,fn in funcs.items():
            for _ in range(warmup):fn()
        # Interleaving reduces systematic bias from testing every model in a fixed long block.
        for iteration in range(repeats):
            for index in rng.permutation(len(order)):
                name=order[index]
                start=time.perf_counter_ns()
                output=funcs[name]()
                elapsed=(time.perf_counter_ns()-start)/1e6
                samples[name].append(elapsed)
            if (iteration+1)%250==0:print('Measured',iteration+1,'repetitions per strategy',flush=True)
    rows=[]
    for name,values in samples.items():
        rows.append(dict(model=name,mean_ms=float(np.mean(values)),median_ms=float(np.median(values)),
                         p90_ms=float(np.quantile(values,.9)),repetitions=repeats))
    result=HERE/'cpu_latency_results'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    result.mkdir(parents=True)
    with (result/'latency_summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (result/'raw_timings.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f);writer.writerow(['iteration',*order])
        writer.writerows([i+1,*[samples[name][i] for name in order]] for i in range(repeats))
    report=dict(cpu=platform.processor(),platform=platform.platform(),torch=torch.__version__,threads=1,
        interop_threads=1,batch=1,device='cpu',precision='float32',record=key,input_shape=list(x.shape),
        duration_s=2.88,warmup=warmup,repetitions=repeats,weights_sha256=hashes,
        waveform_sha256=hashlib.sha256(wave.tobytes()).hexdigest(),inputs_match_original_causal_dataset=True,
        scope='Model execution only. All weights and input already resident in CPU RAM; disk IO, preprocessing and postprocessing excluded. No GPU.',
        streaming='Nine 32-sample stream_step calls, independent zero state per replay, elapsed includes all nine calls. State retained between updates only within each replay.',
        limitation='One waveform repeatedly replayed on this host; timing variability is runtime variation, not variability across earthquake records. Other host workloads may affect results.',rows=rows)
    (result/'benchmark.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (result.parent/'latest_run.json').write_text(json.dumps(dict(directory=str(result)),indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
