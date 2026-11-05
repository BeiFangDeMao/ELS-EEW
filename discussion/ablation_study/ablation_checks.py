"""Short model/loss/gradient/resume/cache checks. Never full-data training."""
import copy
from datetime import datetime
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from ablation_models import build_model,loss,clamp_active,INDICES,Q,resource_stats
from ablation_training import fit
from training import Trainstudents as original
from training_adapter import seed_all,rng_state,restore_rng,configure_original


def run_checks(cfg,specs,waves):
    root=Path(__file__).resolve().parent/'checks'/('synthetic_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    root.mkdir()
    old_device=original.device
    seed_all(42)
    x=torch.randn(2,3,96)
    y=torch.tensor([[4.,50.,0.,1.],[5.,100.,1.,0.]])
    for spec in specs:
        model=build_model(spec)
        if spec['reuse']:
            # Existing PureGRU is constructed through its ORIGINAL class only;
            # no optimizer or backwards update is applied to a reused model.
            continue
        raw=model(x)[0]
        assert raw.shape==(2,3,len(INDICES[spec['task']]))
        value=loss(model,clamp_active(raw,spec['task']),y,spec,cfg['uniform_time_weight'])
        value.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        if not spec['learn_q']:
            assert all(p.grad is None for n,p in model.named_parameters() if 'log_sigma' in n)
        if spec['task']!='all':
            assert [n for n,p in model.named_parameters() if 'log_sigma' in n]==[Q[spec['task']][0]]
    base=build_model(specs[0])
    pred=torch.randn(2,93,4,requires_grad=True)
    original_loss=original.multi_task_weighted_mse_loss(pred,y,base.log_sigma_mag,base.log_sigma_dis,base.log_sigma_azi,T=93)[0]
    adapter_loss=loss(base,pred,y,specs[0],cfg['uniform_time_weight'])
    torch.testing.assert_close(original_loss,adapter_loss,rtol=0,atol=0)
    ga=torch.autograd.grad(original_loss,[pred,base.log_sigma_mag,base.log_sigma_dis,base.log_sigma_azi],retain_graph=True)
    gb=torch.autograd.grad(adapter_loss,[pred,base.log_sigma_mag,base.log_sigma_dis,base.log_sigma_azi])
    for a,b in zip(ga,gb): torch.testing.assert_close(a,b,rtol=0,atol=0)
    # Loss variants must differ only in declared weights.
    uniform=cfg['uniform_time_weight']
    spec=dict(name='test_uniform',task='all',time='uniform',learn_q=True,reuse=None)
    values=(pred-y[:,None,:])**2*uniform
    expected=sum(v/(2*q.exp()**2)+q for v,q in zip(
        [values[:,:,0:1].mean(),values[:,:,1:2].mean(),(values[:,:,2:3].mean()+values[:,:,3:4].mean())/2],
        [base.log_sigma_mag,base.log_sigma_dis,base.log_sigma_azi]))
    torch.testing.assert_close(expected,loss(base,pred,y,spec,uniform),rtol=0,atol=0)
    # Complete training state resume, with a tiny single-task synthetic dataset.
    small=copy.deepcopy(cfg);small.update(epochs=2,batch_size=2,sequence_length=96)
    configure_original(small,torch.device('cpu'))
    task_spec=dict(name='Single_magnitude_SYNTHETIC_ONLY',task='magnitude',time='staged',learn_q=True,reuse=None)
    data=[(torch.randn(3,96),y[i%2],f'synthetic_{i}','00') for i in range(4)]
    def loaders(): return {split:DataLoader(data,batch_size=2,shuffle=split=='train',collate_fn=original.collate_fn) for split in ['train','validation']}
    seed_all(42);model=build_model(task_spec);initial=copy.deepcopy(model.state_dict());initial_rng=rng_state()
    for name in ['continuous','resumed']: (root/name).mkdir()
    try:
        a=build_model(task_spec);a.load_state_dict(initial);restore_rng(initial_rng)
        fit(a,task_spec,small,loaders(),root/'continuous','synthetic',torch.device('cpu'))
        b=build_model(task_spec);b.load_state_dict(initial);restore_rng(initial_rng)
        one=copy.deepcopy(small);one['epochs']=1
        fit(b,task_spec,one,loaders(),root/'resumed','synthetic',torch.device('cpu'))
        fit(build_model(task_spec),task_spec,small,loaders(),root/'resumed','synthetic',torch.device('cpu'))
        a=torch.load(root/'continuous/resume_checkpoint.pth',map_location='cpu',weights_only=False)
        b=torch.load(root/'resumed/resume_checkpoint.pth',map_location='cpu',weights_only=False)
        for key in a['model_state_dict']: torch.testing.assert_close(a['model_state_dict'][key],b['model_state_dict'][key],rtol=0,atol=0)
        assert a['scheduler_state_dict']==b['scheduler_state_dict'] and a['history']==b['history']
        for pid,state in a['optimizer_state_dict']['state'].items():
            for key,value in state.items(): torch.testing.assert_close(value,b['optimizer_state_dict']['state'][pid][key],rtol=0,atol=0)
        # Geometry: zero predicted distance to station/epicenter is zero error;
        # circular stability: 359->1 degrees is a 2-degree change, not 358.
        from ablation_evaluation import evaluate_saved
        toy=np.zeros((1,93,4));angles=np.tile([359,1],47)[:93]
        toy[0,:,2]=np.sin(np.radians(angles));toy[0,:,3]=np.cos(np.radians(angles))
        truth=np.array([[0,0,0,1]],dtype=float)
        metrics,stability=evaluate_saved(toy,truth,specs[0],cfg,np.zeros((1,4)))
        assert np.allclose(metrics.localization_mae,0)
        assert np.allclose(stability[stability.task=='azimuth'].mean_abs_change,2)
        # Cache preflight: original trained8K, 1 real record, 128 samples only.
        from ablation_runner import load_existing
        from cache_benchmark import benchmark
        cache=root/'cache';cache.mkdir()
        benchmark(load_existing(specs[0],cfg)[0],waves[:1,:,:128],cfg,cache,short=True)
        cost=pd.read_csv(cache/'cache_cost_by_update.csv')
        d1=cost[cost['mode']=='D1'].macs_per_update.to_numpy()
        assert np.array_equal(d1,[243296]*4)
        assert np.array_equal(cost[cost['mode']=='D0'].macs_per_update.to_numpy(),243296*np.arange(1,5))
        if cfg['cache']['local_recompute']:
            assert np.array_equal(cost[cost['mode']=='D2'].macs_per_update.to_numpy(),[243296,281216,281216,281216])
        # Actual archived predictions/metrics only for plotting smoke checks.
        from ablation_reporting import make_plots
        actual=[]
        archives=[('ELS8K_Full',str(Path(cfg['reference_checkpoint']).parent/'test_metrics_by_chunk.json'))]
        if cfg['groups']['architecture']: archives.append(('PureGRU_Existing',cfg['pure_metrics']))
        for name,path in archives:
            archive=json.loads(Path(path).read_text(encoding='utf-8'))
            for j,item in enumerate(archive['by_chunk']):
                m=item['metrics'];actual.append(dict(model=name,observation_time_s=(j+1)*.32,
                    magnitude_mae=m['mag']['mae'],distance_mae=m['dis']['mae'],azimuth_mae=m['azi_deg']['mae']))
        actual=pd.DataFrame(actual)
        resources=pd.read_csv(Path(__file__).resolve().parent/'checks/model_resources.csv')
        latency=pd.read_csv(cache/'cache_efficiency.csv')
        make_plots(root,actual,resources,actual[actual.model=='ELS8K_Full'],latency,write_files=False)
        return dict(passed=True,all_new_models_forward_backward=True,baseline_loss_and_gradients_identical=True,
            frozen_q_has_no_grad=True,single_task_has_no_unrelated_outputs_or_q=True,
            complete_resume_bitwise_equal_cpu=True,circular_change_359_to_1_is_2=True,spherical_zero_error_checked=True,
            cache_short_consistency=True,incremental_macs_hook_verified=True,
            actual_archived_results_plot_smoke_check=True,
            real_training_started=False,synthetic_records=4,synthetic_length=96,output_directory=str(root))
    finally: configure_original(cfg,old_device)
