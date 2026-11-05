"""Tiny synthetic validation of real training/resume code, never formal training."""
from datetime import datetime
import copy
import json
import logging
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from training import Trainstudents as original
from model_scaling import make_model
from training_adapter import (atomic_torch,seed_all,configure_original,optimizer_scheduler,
                              train_model,update_selection,rng_state,restore_rng)
from reporting import nondominated,pareto_tables,recommendations


def run_checks(cfg):
    root=Path(__file__).resolve().parent/'checks'/'synthetic_artifacts'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    root.mkdir(parents=True)
    logging.info('Synthetic checks only: 4 records, 96 points, at most two tiny CPU epochs per path. NOT real-data training.')
    small=copy.deepcopy(cfg)
    small.update(sequence_length=96,batch_size=2,epochs=2)
    seed_all(42)
    waves=torch.randn(4,3,96)
    targets=torch.tensor([[4.,50.,0.,1.],[5.,60.,1.,0.],[4.5,70.,0.,-1.],[6.,100.,-1.,0.]])
    samples=[(waves[i],targets[i],f'synthetic_{i}','00') for i in range(4)]
    loader=lambda shuffle: DataLoader(samples,batch_size=2,shuffle=shuffle,collate_fn=original.collate_fn)
    spec=dict(model='SYNTHETIC_ONLY',cnn_width=4,gru_hidden=6)
    configure_original(small,torch.device('cpu'))
    seed_all(42)
    base=make_model(spec)
    initial=copy.deepcopy(base.state_dict())
    initial_rng=rng_state()
    dirs={name:root/name for name in ('continuous','resumed')}
    for path in dirs.values(): path.mkdir()
    try:
        # Restore must include RNG and scheduler, not merely model weights.
        uninterrupted=make_model(spec);uninterrupted.load_state_dict(initial)
        restore_rng(initial_rng)
        train_model(uninterrupted,spec,small,(loader(True),loader(False)),dirs['continuous'],'synthetic')
        first=make_model(spec);first.load_state_dict(initial)
        restore_rng(initial_rng)
        one=copy.deepcopy(small);one['epochs']=1
        train_model(first,spec,one,(loader(True),loader(False)),dirs['resumed'],'synthetic')
        resumed=make_model(spec)
        train_model(resumed,spec,small,(loader(True),loader(False)),dirs['resumed'],'synthetic')
        a=torch.load(dirs['continuous']/'resume_checkpoint.pth',map_location='cpu',weights_only=False)
        b=torch.load(dirs['resumed']/'resume_checkpoint.pth',map_location='cpu',weights_only=False)
        for key in a['model_state_dict']:
            torch.testing.assert_close(a['model_state_dict'][key],b['model_state_dict'][key],rtol=0,atol=0)
        assert a['scheduler_state_dict']==b['scheduler_state_dict']
        assert a['history']==b['history']
        assert a['epoch']==b['epoch']==2
        assert a['optimizer_state_dict']['param_groups']==b['optimizer_state_dict']['param_groups']
        for pid,sa in a['optimizer_state_dict']['state'].items():
            for key,value in sa.items():
                other=b['optimizer_state_dict']['state'][pid][key]
                if isinstance(value,torch.Tensor): torch.testing.assert_close(value,other,rtol=0,atol=0)
                else: assert value==other
        best,bad,accept=update_selection(np.array([1.,2.,3.]),np.array([.9,2.1,2.9]),7,1e-6)
        assert accept and bad==0 and best.tolist()==[.9,2.,2.9]
        best2,bad,accept=update_selection(best,np.array([.8,2.1,3.]),0,1e-6)
        assert not accept and bad==1 and np.array_equal(best,best2)
        # Explicitly verify failure, not silent skipping, on an invalid batch.
        invalid=[(torch.full((3,96),float('nan')),targets[0],'synthetic_nan','00')]
        invalid_loader=DataLoader(invalid,batch_size=1,collate_fn=original.collate_fn)
        path=root/'nonfinite';path.mkdir()
        rejected=False
        try:
            train_model(make_model(spec),spec,one,(invalid_loader,loader(False)),path,'synthetic')
        except FloatingPointError:
            rejected=True
        assert rejected and not (path/'resume_checkpoint.pth').exists()
        assert nondominated([[1,3],[2,4],[3,1],[1,3]]).tolist()==[True,False,True,True]
        # Pure reporting logic tests use only algebra; no fake analysis figures.
        import pandas as pd
        costs=pd.DataFrame([dict(model='tiny',total_parameters=2,macs_per_update=3,persistent_state_kib=1),
                            dict(model='wide',total_parameters=4,macs_per_update=5,persistent_state_kib=2)])
        values=[]
        for split in ['validation','test']:
            for time in [3,20]:
                for model in ['tiny','wide']:
                    error=(1 if model=='tiny' else 2) if split=='validation' else (2 if model=='tiny' else 1)
                    values.append(dict(model=model,split=split,requested_time_s=time,actual_output_time_s=2.88 if time==3 else 19.84,
                                       magnitude_mae=error,distance_mae_km=error,azimuth_circular_mae_deg=error))
        values=pd.DataFrame(values)
        front=pareto_tables(values,costs)
        choices=recommendations(values,costs,dict(budgets=[dict(name='unit',max_parameters=10,max_macs=10,max_state_kib=10)]))
        assert set(choices.model)=={'tiny'}, 'Test performance must not affect recommendations'
        # Render all plot types with ACTUAL archived 8K test statistics only.
        # No file export and no synthetic values enter the plotting pipeline.
        from reporting import figures
        source=Path(cfg['reference_checkpoint']).parent/'test_metrics_by_chunk.json'
        archive=json.loads(source.read_text(encoding='utf-8'))['by_chunk']
        actual=[]
        for requested,available in [(3,2.88),(5,4.8),(10,9.92),(20,19.84),(29.76,29.76)]:
            index=int(np.floor(available/.32+1e-9))-1
            part=archive[index]['metrics']
            actual.append(dict(model='ELS-8K',split='test',requested_time_s=requested,actual_output_time_s=available,
                magnitude_mae=part['mag']['mae'],distance_mae_km=part['dis']['mae'],
                azimuth_circular_mae_deg=part['azi_deg']['mae']))
        actual=pd.DataFrame(actual)
        actual_resources=pd.read_csv(Path(__file__).resolve().parent/'checks/compute_resources.csv')
        actual_front=pareto_tables(actual,actual_resources)
        rendered=figures(root/'actual_8K_plot_checks',actual,actual_resources,actual_front,write_files=False)
        assert rendered==6
        # Restore a checkpoint at its terminal epoch: it must not train again.
        train_model(make_model(spec),spec,small,(loader(True),loader(False)),dirs['resumed'],'synthetic')
        after=torch.load(dirs['resumed']/'resume_checkpoint.pth',map_location='cpu',weights_only=False)
        assert after['epoch']==2
        return dict(passed=True,synthetic_only=True,real_training_started=False,
                    resume_bitwise_equal_to_uninterrupted_cpu=True,scheduler_optimizer_rng_restored=True,
                    original_selection_semantics=True,nonfinite_batch_aborts=True,pareto_ties_checked=True,
                    recommendations_ignore_test_performance=True,terminal_resume_does_not_retrain=True,
                    six_actual_data_plot_canvases_rendered_without_export=True,
                    output_directory=str(root))
    finally:
        configure_original(cfg,torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
