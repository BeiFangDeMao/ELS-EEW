"""Fixed 16-channel, two-layer hidden-24 backbone. Reuse existing PureGRU."""
import torch
from torch import nn
from models.compact_gru import CompactGRU
from models.pure_gru import PureGRU
from training import Trainstudents as original

Q={'magnitude':('log_sigma_mag',-1.),'distance':('log_sigma_dis',4.),'azimuth':('log_sigma_azi',-1.5)}
INDICES={'magnitude':[0],'distance':[1],'azimuth':[2,3],'all':[0,1,2,3]}


def experiments(cfg):
    result=[dict(name='ELS8K_Full',task='all',time='staged',learn_q=True,reuse='reference')]
    if cfg['groups']['architecture']:
        result.append(dict(name='PureGRU_Existing',task='all',time='staged',learn_q=True,reuse='pure'))
    if cfg['groups']['multitask']:
        for task in ['magnitude','distance','azimuth']:
            result.append(dict(name='Single_'+task,task=task,time='staged',learn_q=cfg['single_task_learn_own_q'],reuse=None))
    if cfg['groups']['loss']:
        for name,time,learn in [('C0_Uniform_Fixed','uniform',False),('C1_Staged_Fixed','staged',False),('C2_Uniform_Learned','uniform',True)]:
            result.append(dict(name=name,task='all',time=time,learn_q=learn,reuse=None))
    return result


def build_model(spec):
    if spec.get('reuse')=='pure': return PureGRU()
    model=CompactGRU()
    task=spec['task']
    if task!='all':
        model.output_head.proj=nn.Linear(24,len(INDICES[task]))
        for name,_ in Q.values():
            if name!=Q[task][0]: delattr(model,name)
    for name,param in model.named_parameters():
        if 'log_sigma' in name: param.requires_grad_(spec['learn_q'])
    return model


def clamp_active(pred,task):
    if task=='all': return original.safe_clamp_preds(pred)
    if task=='magnitude': return pred.clamp(0,10)
    if task=='distance': return pred.clamp(0,200)
    return pred.clamp(-1,1)


def time_weights(pred,mode,uniform):
    n=pred.shape[1]
    if mode=='uniform': return pred.new_full((1,n,1),uniform)
    stage=(torch.arange(n,device=pred.device)*4//n).clamp_max(3)
    return pred.new_tensor([.7,.8,.9,1.])[stage][None,:,None]


def loss(model,pred,y,spec,uniform=.85):
    if spec['task']=='all' and spec['time']=='staged':
        return original.multi_task_weighted_mse_loss(pred,y,model.log_sigma_mag,model.log_sigma_dis,model.log_sigma_azi,T=pred.shape[1])[0]
    weights=time_weights(pred,spec['time'],uniform)
    target=y[:,None,INDICES[spec['task']]].expand(-1,pred.shape[1],-1)
    if spec['task']=='all':
        values=((pred-target)**2*weights)
        raws=[values[:,:,0:1].mean(),values[:,:,1:2].mean(),(values[:,:,2:3].mean()+values[:,:,3:4].mean())/2]
        qs=[model.log_sigma_mag,model.log_sigma_dis,model.log_sigma_azi]
    else:
        raw=((pred-target)**2*weights).mean()
        raws=[raw];qs=[getattr(model,Q[spec['task']][0])]
    return sum(raw/(2*torch.exp(q)**2)+q for raw,q in zip(raws,qs))


def task_maes(pred,truth,task):
    """Original metric definitions on flattened float32 outputs/labels."""
    import numpy as np
    if task=='all':
        metrics=original.compute_task_metrics(pred,truth)
        return [float(metrics['mag']['mae']),float(metrics['dis']['mae']),float(metrics['azi_deg']['mae'])]
    target=truth[:,INDICES[task]]
    if task!='azimuth': return [float(np.mean(np.abs(pred-target)))]
    pa=np.degrees(np.arctan2(pred[:,0],pred[:,1]))%360
    ta=np.degrees(np.arctan2(target[:,0],target[:,1]))%360
    return [float(np.mean(np.minimum(np.abs(pa-ta),360-np.abs(pa-ta))))]


def resource_stats(model,name):
    """Count actual increment, including every GRU timestep; no nonexistent CNN group."""
    count=lambda module:sum(p.numel() for p in module.parameters())
    counters=dict(cnn=0,gru=0,output=0);handles=[]
    def conv(mod,inp,out): counters['cnn']+=out.numel()*(mod.in_channels//mod.groups)*mod.kernel_size[0]
    def gru(mod,inp,out):
        for layer in range(mod.num_layers):
            ins=mod.input_size if layer==0 else mod.hidden_size
            counters['gru']+=inp[0].shape[0]*inp[0].shape[1]*3*mod.hidden_size*(ins+mod.hidden_size)
    def linear(mod,inp,out): counters['output']+=out.numel()*mod.in_features
    for mod in model.modules():
        if isinstance(mod,nn.Conv1d): handles.append(mod.register_forward_hook(conv))
        elif isinstance(mod,nn.GRU): handles.append(mod.register_forward_hook(gru))
        elif isinstance(mod,nn.Linear): handles.append(mod.register_forward_hook(linear))
    model.eval()
    try:
        with torch.inference_mode(): _,state=model.stream_step(torch.zeros(1,3,32,device=next(model.parameters()).device))
    finally:
        for hook in handles: hook.remove()
    def numel(value):
        if isinstance(value,torch.Tensor): return value.numel()
        if isinstance(value,(tuple,list)): return sum(numel(x) for x in value)
        return 0
    pure=isinstance(model,PureGRU)
    conv_state=0 if pure else numel(state[0])
    gru_state=numel(state) if pure else numel(state[1])
    return dict(model=name,total_parameters=count(model),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                cnn_parameters=count(model.feature_extractor),gru_parameters=count(model.output_head.gru),
                output_parameters=count(model.output_head.proj),cnn_macs=counters['cnn'],gru_macs=counters['gru'],output_macs=counters['output'],
                macs_per_update=sum(counters.values()),fp32_weights_kib=count(model)*4/1024,
                conv_cache_kib=conv_state*4/1024,gru_hidden_kib=gru_state*4/1024,persistent_state_kib=(conv_state+gru_state)*4/1024,
                temporary_activation_memory='not measured',parameter_match_to_8k=abs(count(model)-8263)/8263<=.05)
