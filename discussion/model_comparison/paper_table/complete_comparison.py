"""Summarize saved test predictions and count dense MACs at batch=1 on CPU."""
from pathlib import Path
import csv,json,sys
import numpy as np
import torch
from torch import nn
BASE=Path(__file__).resolve().parents[1]
ROOT=BASE.parents[1]
sys.path.insert(0,str(BASE))
from train_comparison import make_model

def r2(y,p):
    return float(1-np.square(p-y).sum()/np.square(y-y.mean(axis=0)).sum())

def profile(model):
    counts=dict(conv=0,linear=0,lstm=0,attention=0)
    handles=[]
    def conv(m,x,y):counts['conv']+=y.numel()*(m.in_channels//m.groups)*m.kernel_size[0]
    def linear(m,x,y):counts['linear']+=y.numel()*m.in_features
    def lstm(m,x,y):
        batch,steps=x[0].shape[:2]
        directions=2 if m.bidirectional else 1
        for layer in range(m.num_layers):
            ins=m.input_size if layer==0 else m.hidden_size*directions
            counts['lstm']+=batch*steps*directions*4*m.hidden_size*(ins+m.hidden_size)
    def attention(m,x,y):
        b,c,l=x[0].shape
        counts['attention']+=2*b*c*l*m._profile_k_length
    for m in model.modules():
        if isinstance(m,nn.Conv1d):handles.append(m.register_forward_hook(conv))
        elif isinstance(m,nn.Linear):handles.append(m.register_forward_hook(linear))
        elif isinstance(m,nn.LSTM):handles.append(m.register_forward_hook(lstm))
        if type(m).__name__=='AttentionBlock':
            handles.append(m.k_proj.register_forward_hook(lambda mod,x,y,parent=m:setattr(parent,'_profile_k_length',y.shape[-1])))
            handles.append(m.register_forward_hook(attention))
    with torch.inference_mode():model(torch.zeros(1,3,288))
    for h in handles:h.remove()
    return dict(parameters=sum(p.numel() for p in model.parameters()),macs=sum(counts.values()),breakdown=counts)

def main():
    torch.set_num_threads(2)
    run=BASE/'training_results/20261022_212803_440932'
    cfg=json.loads((run/'identity.json').read_text())['config']
    result={}
    for name in ['seist_s_emg','seist_s_dis','seist_s_baz','magnet']:
        with (run/name/'test_predictions.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
        assert len(rows)==51772
        y=np.array([float(r['truth']) for r in rows]);p=np.array([float(r['prediction']) for r in rows])
        if name.endswith('baz'):
            error=abs((p-y+180)%360-180)
            yy=np.column_stack([np.sin(np.deg2rad(y)),np.cos(np.deg2rad(y))])
            pp=np.column_stack([np.sin(np.deg2rad(p)),np.cos(np.deg2rad(p))])
            score=r2(yy,pp)
        else:error=abs(p-y);score=r2(y,p)
        computed=dict(mae=float(error.mean()),r2=score,records=len(y),**profile(make_model(name,cfg).eval()))
        saved=json.loads((run/name/'completed.json').read_text())
        assert abs(computed['mae']-saved['mae'])<1e-5
        assert computed['parameters']==saved['parameters']
        result[name]=computed
    source=ROOT/'discussion/early_reliability_warning/results/20261018_180137_947801_record_only/predictions'
    pred=np.load(source/'processed_predictions.npy',mmap_mode='r')[:,8,:].astype('float64')
    y=np.load(source/'truths.npy',mmap_mode='r')
    theta=np.deg2rad(y[:,2]);yz=np.column_stack([np.sin(theta),np.cos(theta)])
    result['ELS-EEW']=dict(magnitude_r2=r2(y[:,0],pred[:,0]),distance_r2=r2(y[:,1],pred[:,1]),baz_r2=r2(yz,pred[:,2:4]),
        parameters=8263,macs_per_32_sample_update=32*(4*16*16+15*16)+32*(3*24*16+9*24*24)+4*24)
    result['protocol']=dict(time_s=2.88,records=51772,
        macs='Dense multiply-accumulates only; conv/linear/recurrent gates/attention QK and AV. Bias, normalization, activations, softmax and pooling excluded. CPU batch=1 shape-based actual forward hooks; no training or timing.',
        baz_r2='Pooled sin/cos SSE divided by pooled SST. SeisT angle converted to unit vector; ELS original stored clipped sin/cos, without normalization.',
        seisT='Three separately trained single-task models; resource costs must be summed for all three tasks.',
        warning='ELS per-update MACs cover 32 new samples; SeisT/MagNet MACs cover the full available 288-sample window. These are not equal-duration cumulative computation costs.')
    (Path(__file__).parent/'comparison_values.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
