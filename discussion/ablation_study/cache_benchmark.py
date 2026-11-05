"""Numerically checked D0/D1/D2, measured batch-1 CPU latency, no file I/O in timers."""
import copy
import platform
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm
from training_adapter import atomic_json


def raw_history(model):
    # Longest residual path: sum dilation*(kernel-1), not a feature cache.
    return sum(block.convs[0].history for block in model.feature_extractor.blocks)


def reset(model,mode):
    if mode=='D0': return None
    reference=next(model.parameters())
    hidden=reference.new_zeros(2,1,24)
    if mode=='D1':
        front=tuple((reference.new_zeros(1,16,b.convs[0].history),) for b in model.feature_extractor.blocks)
        return front,(hidden,None)
    return reference.new_zeros(1,3,0),(hidden,None)


def step(model,mode,wave,chunk_index,state):
    end=(chunk_index+1)*32
    if mode=='D0': return model(wave[:,:,:end])[0][:,-1:],None
    chunk=wave[:,:,end-32:end]
    if mode=='D1': return model.stream_step(chunk,state)
    context,head_state=state
    received=torch.cat([context,chunk],dim=-1)
    features,_=model.feature_extractor(received)
    prediction,new_head=model.output_head(features[:,:,-32:],chunk,head_state)
    return prediction,(received[:,:,-raw_history(model):].clone(),new_head)


def tensor_values(state):
    if isinstance(state,torch.Tensor): return state.numel()
    if isinstance(state,(list,tuple)): return sum(tensor_values(item) for item in state)
    return 0


def mac_hooks(model):
    counts={'value':0};handles=[]
    def conv(m,x,y): counts['value']+=y.numel()*(m.in_channels//m.groups)*m.kernel_size[0]
    def gru(m,x,y):
        for layer in range(m.num_layers):
            ins=m.input_size if layer==0 else m.hidden_size
            counts['value']+=x[0].shape[0]*x[0].shape[1]*3*m.hidden_size*(ins+m.hidden_size)
    def linear(m,x,y): counts['value']+=y.numel()*m.in_features
    for module in model.modules():
        if isinstance(module,nn.Conv1d): handles.append(module.register_forward_hook(conv))
        elif isinstance(module,nn.GRU): handles.append(module.register_forward_hook(gru))
        elif isinstance(module,nn.Linear): handles.append(module.register_forward_hook(linear))
    return counts,handles


def benchmark(model,waves,cfg,folder,short=False):
    model=copy.deepcopy(model).cpu().float().eval()
    if waves.device.type!='cpu': waves=waves.cpu()
    waves=waves.float().contiguous()
    if waves.shape[1]!=3 or waves.shape[2]%32: raise ValueError('Invalid cache inputs')
    if raw_history(model)!=30: raise ValueError('Unexpected 8K receptive field')
    settings=cfg['cache'];steps=waves.shape[-1]//32
    modes=['D0','D1']+(['D2'] if settings['local_recompute'] else [])
    previous=torch.get_num_threads();torch.set_num_threads(settings['threads'])
    consistency=[];costs=[];delays={mode:[] for mode in modes};resets={mode:[] for mode in modes}
    try:
        with torch.inference_mode():
            # Record-by-record reset; compare raw, unclipped four outputs.
            reference=model(waves)[0]
            for mode in modes:
                outputs=[]
                for wave in waves.split(1):
                    state=reset(model,mode);preds=[]
                    for j in range(steps):
                        pred,state=step(model,mode,wave,j,state);preds.append(pred)
                    outputs.append(torch.cat(preds,dim=1))
                outputs=torch.cat(outputs)
                torch.testing.assert_close(reference,outputs,rtol=2e-4,atol=2e-5)
                delta=(reference-outputs).abs()
                for i,name in enumerate(['magnitude','distance_km','sin','cos']):
                    consistency.append(dict(mode=mode,output=name,max_absolute_difference=float(delta[:,:,i].max()),
                                            mean_absolute_difference=float(delta[:,:,i].mean()),records=len(waves),updates=steps,
                                            passed=True,rtol=2e-4,atol=2e-5))
                counts,handles=mac_hooks(model)
                try:
                    state=reset(model,mode)
                    for j in range(steps):
                        counts['value']=0
                        _,state=step(model,mode,waves[:1],j,state)
                        if mode=='D0':
                            model_values=0;raw_values=3*32*(j+1)
                        elif mode=='D1':
                            model_values=tensor_values(state);raw_values=0
                        else:
                            model_values=tensor_values(state[1]);raw_values=state[0].numel()
                        costs.append(dict(mode=mode,chunk=j+1,observation_time_s=(j+1)*.32,macs_per_update=counts['value'],
                                          model_state_kib=model_values*4/1024,input_history_kib=raw_values*4/1024,
                                          total_persistent_history_kib=(model_values+raw_values)*4/1024,
                                          repeats_history_computation=mode!='D1'))
                finally:
                    for handle in handles: handle.remove()
            # All weights, inputs and reference computations are outside timing.
            for mode in modes:
                warmups=0 if short else settings['warmup_records']
                repeats=1 if short else settings['repeats']
                for i in range(warmups):
                    wave=waves[i%len(waves):i%len(waves)+1];state=reset(model,mode)
                    for j in range(steps): _,state=step(model,mode,wave,j,state)
                for repeat in tqdm(range(repeats),desc=f'CPU cache {mode}, threads={settings["threads"]}'):
                    for record,wave in enumerate(waves.split(1)):
                        start=time.perf_counter_ns();state=reset(model,mode)
                        resets[mode].append((time.perf_counter_ns()-start)/1e6)
                        measured=[]
                        for j in range(steps):
                            start=time.perf_counter_ns()
                            _,state=step(model,mode,wave,j,state)
                            measured.append((time.perf_counter_ns()-start)/1e6)
                        # Logging and list bookkeeping happen after timed calls.
                        delays[mode].append(measured)
    finally: torch.set_num_threads(previous)
    resource=pd.DataFrame(costs)
    rows=[];by_update=[]
    for mode in modes:
        data=np.asarray(delays[mode]);totals=data.sum(axis=1)
        current=resource[resource['mode']==mode]
        rows.append(dict(mode=mode,median_update_ms=float(np.median(data)),p95_update_ms=float(np.percentile(data,95)),
                         median_record_total_ms=float(np.median(totals)),p95_record_total_ms=float(np.percentile(totals,95)),
                         median_initialization_ms=float(np.median(resets[mode])),
                         total_macs=int(current.macs_per_update.sum()),first_update_macs=int(current.macs_per_update.iloc[0]),
                         final_update_macs=int(current.macs_per_update.iloc[-1]),
                         final_model_state_kib=float(current.model_state_kib.iloc[-1]),
                         final_input_history_kib=float(current.input_history_kib.iloc[-1]),
                         final_total_history_kib=float(current.total_persistent_history_kib.iloc[-1]),
                         records=len(waves),repeats=1 if short else settings['repeats'],updates=steps,threads=settings['threads'],
                         batch_size=1,scope='pure model/update; no disk I/O, preprocessing, logging or plotting',
                         update_p95_below_320ms=bool(np.percentile(data,95)<320)))
        for j in range(steps):
            by_update.append(dict(mode=mode,chunk=j+1,observation_time_s=(j+1)*.32,
                                 median_update_ms=float(np.median(data[:,j])),p95_update_ms=float(np.percentile(data[:,j],95))))
        np.save(folder/f'{mode}_latency_samples_ms.npy',data)
    summary=pd.DataFrame(rows)
    d0=summary.set_index('mode').loc['D0']
    summary['record_time_speedup_vs_D0']=d0.median_record_total_ms/summary.median_record_total_ms
    summary['total_macs_reduction_vs_D0']=d0.total_macs/summary.total_macs
    pd.DataFrame(consistency).to_csv(folder/'cache_consistency.csv',index=False)
    summary.to_csv(folder/'cache_efficiency.csv',index=False)
    resource.to_csv(folder/'cache_cost_by_update.csv',index=False)
    pd.DataFrame(by_update).to_csv(folder/'cache_latency_by_update.csv',index=False)
    atomic_json(folder/'environment.json',dict(cpu=platform.processor(),platform=platform.platform(),torch=torch.__version__,
        torch_threads=settings['threads'],mkldnn_enabled=torch.backends.mkldnn.enabled,fp32=True,batch_size=1,
        short_preflight=short,raw_context_samples=30,
        note='D0 input buffer grows with prefix; D2 stores raw context, not hidden convolution caches. Reset allocation timed separately.'))
    return summary
