# -*- coding: utf-8 -*-
"""8K/16K/32K paired localization evaluation on the identical independent test.

Inference only. Reuse original test-record reader, output clipping and spherical
geometry. Save all 93 predictions/errors so other thresholds need no inference.
"""
from datetime import datetime
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

HERE=Path(__file__).resolve().parent
STUDY=HERE.parent
WORKSPACE=STUDY.parents[1]
sys.path.insert(0,str(WORKSPACE/'els_eew'))
sys.path.insert(0,str(STUDY))
sys.path.insert(0,str(WORKSPACE/'discussion/early_reliability_warning'))
from model_scaling import make_model,validate_streaming
from inference import TestData,selected_keys,destination,separation,clamp


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    active=json.loads((STUDY/'active_run.json').read_text(encoding='utf-8'))
    run=STUDY/'results'/active['run_directory_name']
    source=json.loads((run/'experiment_identity.json').read_text(encoding='utf-8'))['config']
    paths={
        'ELS-8K':Path(source['reference_checkpoint']),
        'ELS-16K':run/'ELS-16K/best.pth',
        'ELS-32K':run/'ELS-32K/best.pth',
    }
    specs={'ELS-8K':dict(cnn_width=16,gru_hidden=24),'ELS-16K':dict(cnn_width=22,gru_hidden=34),
           'ELS-32K':dict(cnn_width=34,gru_hidden=48)}
    keys,_=selected_keys()
    assert len(keys)==51772
    out=HERE/'results'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(4)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False
    models={}; provenance={}
    for name,path in paths.items():
        state=json.loads((run/name/'state.json').read_text(encoding='utf-8'))
        assert state['status']=='complete',f'{name} is not complete'
        for filename,expected in state['output_hashes'].items():
            assert digest(run/name/filename)==expected,filename
        checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        model=make_model(specs[name])
        model.load_state_dict(checkpoint['model_state_dict'],strict=True)
        model.to(device).eval()
        check=validate_streaming(model,torch.zeros(1,3,2976,device=device))
        models[name]=model
        provenance[name]=dict(checkpoint=str(path),sha256=digest(path),epoch=checkpoint['epoch'],stream_check=check)
    n,t=len(keys),93
    predictions={name:np.lib.format.open_memmap(out/f'{name}_predictions.npy',mode='w+',dtype=np.float32,shape=(n,t,4)) for name in models}
    errors={name:np.lib.format.open_memmap(out/f'{name}_location_errors_km.npy',mode='w+',dtype=np.float64,shape=(n,t)) for name in models}
    truth=np.lib.format.open_memmap(out/'truths.npy',mode='w+',dtype=np.float64,shape=(n,7))
    np.save(out/'record_keys.npy',keys,allow_pickle=False)
    data=TestData(keys)
    loader=DataLoader(data,batch_size=512,shuffle=False,num_workers=0)
    offset=0
    with torch.inference_mode():
        for wave,y,_ in tqdm(loader,desc='8K/16K/32K localization; same batch for all models'):
            values=y.numpy()
            sl=slice(offset,offset+len(values))
            truth[sl]=values
            la,lo=np.radians(values[:,3:4]),np.radians(values[:,4:5])
            true_la,true_lo=np.radians(values[:,5:6]),np.radians(values[:,6:7])
            x=wave.to(device)
            for name,model in models.items():
                raw=model(x)[0]
                assert raw.shape==(len(values),93,4) and torch.isfinite(raw).all()
                p=clamp(raw.cpu().numpy())
                predictions[name][sl]=p
                # atan2(sin,cos): station-to-epicenter bearing, clockwise from N.
                bearing=np.arctan2(p[:,:,2].astype(np.float64),p[:,:,3].astype(np.float64))
                predicted_la,predicted_lo=destination(la,lo,p[:,:,1].astype(np.float64),bearing)
                error=separation(true_la,true_lo,predicted_la,predicted_lo)
                assert np.isfinite(error).all() and (error>=0).all()
                errors[name][sl]=error
            offset+=len(values)
    assert offset==n
    if data.file is not None: data.file.close()
    truth.flush()
    for arr in [*predictions.values(),*errors.values()]: arr.flush()
    times=np.arange(1,94)*.32
    groups={'all':np.ones(n,dtype=bool)}
    for d in [10,50,100,150]: groups[f'D_le_{d}']=truth[:,1]<=d
    for lo,hi in [(0,10),(10,50),(50,100),(100,200)]:
        groups[f'D_{lo}_{hi}']=(truth[:,1]>=lo)&(truth[:,1]<hi)
    rows=[]
    for name,error in errors.items():
        for group,mask in groups.items():
            e=np.asarray(error[mask]);count=int(mask.sum())
            median,p90=np.percentile(e,[50,90],axis=0,method='linear')
            for j,time in enumerate(times):
                row=dict(model=name,group=group,records=count,observation_time_s=time,
                         mean_location_error_km=float(e[:,j].mean()),median_location_error_km=float(median[j]),
                         p90_location_error_km=float(p90[j]))
                for threshold in [5,10,20,30,50]: row[f'within_{threshold}km']=float((e[:,j]<=threshold).mean())
                rows.append(row)
    metrics=pd.DataFrame(rows)
    metrics.to_csv(out/'localization_by_time.csv',index=False,encoding='utf-8-sig')
    targets=[3,5,10,20,29.76]
    actual=[(int(np.floor(v/.32+1e-9)))*.32 for v in targets]
    selected=metrics[metrics.observation_time_s.isin(actual)].copy()
    selected['requested_time_s']=selected.observation_time_s.map(dict(zip(actual,targets)))
    selected.to_csv(out/'localization_key_times.csv',index=False,encoding='utf-8-sig')
    paired=[]
    for a,b in [('ELS-8K','ELS-16K'),('ELS-8K','ELS-32K'),('ELS-16K','ELS-32K')]:
        for group,mask in groups.items():
            for target,time in zip(targets,actual):
                index=int(round(time/.32))-1
                ea,eb=errors[a][mask,index],errors[b][mask,index]
                reduction=ea.mean()-eb.mean()
                paired.append(dict(smaller=a,larger=b,group=group,records=len(ea),requested_time_s=target,
                    actual_output_time_s=time,smaller_mean_km=float(ea.mean()),larger_mean_km=float(eb.mean()),
                    mean_reduction_km=float(reduction),relative_reduction_pct=float(100*reduction/ea.mean()),
                    larger_model_better_fraction=float((eb<ea).mean())))
    paired=pd.DataFrame(paired)
    paired.to_csv(out/'paired_comparison.csv',index=False,encoding='utf-8-sig')
    # Verify source-independent geometry and uniform inference against saved
    # scaling metrics (same 512-record batch size and postprocessing).
    previous=pd.read_csv(run/'accuracy_by_time.csv')
    checks=[]
    for name,p in predictions.items():
        for target,time in zip(targets,actual):
            j=int(round(time/.32))-1
            angle=np.degrees(np.arctan2(p[:,j,2].astype(np.float64),p[:,j,3].astype(np.float64)))%360
            da=np.abs((angle-truth[:,2]+180)%360-180)
            dis=np.abs(p[:,j,1].astype(np.float64)-truth[:,1]).mean()
            prior=previous[(previous.model==name)&(previous.split=='test')&(previous.requested_time_s==target)].iloc[0]
            assert abs(dis-prior.distance_mae_km)<1e-3
            assert abs(da.mean()-prior.azimuth_circular_mae_deg)<1e-3
            checks.append(dict(model=name,time_s=time,distance_mae_difference=float(dis-prior.distance_mae_km),
                               angle_maae_difference=float(da.mean()-prior.azimuth_circular_mae_deg)))
    for name,path in paths.items(): assert digest(path)==provenance[name]['sha256']
    manifest=dict(source_experiment=str(run),source_dataset=source['dataset'],records=n,aggregation='waveform_record',
        time_points=t,batch_size=512,spherical_radius_km=6371,geometry='Original Experiment01 destination + separation',
        clipping='M [0,10], D [0,200], sin/cos [-1,1]',models=provenance,
        cuda_cudnn_tf32=torch.backends.cudnn.allow_tf32,training_performed=False,
        validation_checks_against_scaling_metrics=checks,
        truth_columns=['magnitude','distance_km','bearing_deg','station_lat','station_lon','epicenter_lat','epicenter_lon'])
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 8K、16K、32K定位误差补充分析',
           '对同一独立测试集51772条波形重新加载最佳权重推理，不训练、不微调。三个模型共用每一批相同波形和时间索引。',
           '定位误差为预测震中与真实目录震中的球面大圆距离，单位km；平均定位误差是非负空间误差的均值，不是经纬度坐标分量的MAE。',
           '距离及方位角后处理、R=6371 km球面正向定位和距离函数均复用Experiment01。全部93个时间点保留。',
           '\n## 全测试集关键时刻',selected[selected.group=='all'][['model','requested_time_s','observation_time_s','mean_location_error_km','median_location_error_km','p90_location_error_km','within_10km','within_20km','within_30km']].to_csv(index=False),
           '\n## 配对比较',paired[paired.group=='all'].to_csv(index=False),
           '\n## 按真实震中距分组的20秒结果（实际19.84秒）',
           selected[(selected.requested_time_s==20)&selected.group.isin(['D_le_10','D_le_50','D_0_10','D_10_50','D_50_100','D_100_200'])].to_csv(index=False),
           '\n## 解释边界','所有比例均为波形记录级，不是独立事件数或事件级置信度；单随机种子描述性比较，不作显著性结论。',
           '更大模型的平均误差下降不代表每条记录都改善，逐记录改善比例见paired_comparison.csv。',
           '本分析是独立测试性能报告，不用这些测试结果挑选最终模型；原有验证集与预算选择规则保持不变。',
           '权重SHA256核对前后未变；距离和角度误差已与规模实验相同测试时刻逐项核对。']
    (out/'summary.md').write_text('\n\n'.join(lines),encoding='utf-8')
    (HERE/'latest_result.json').write_text(json.dumps(dict(result_directory=str(out)),ensure_ascii=False,indent=2),encoding='utf-8')
    # Add analysis to the existing report without changing its original tables.
    supplement=run/'localization_comparison.md'
    supplement.write_text('\n\n'.join(lines),encoding='utf-8')
    summary=run/'summary.md'
    text=summary.read_text(encoding='utf-8')
    marker='## 附加：8K、16K、32K定位误差'
    if marker not in text:
        summary.write_text(text+'\n\n'+marker+'\n\n见[定位误差补充分析](localization_comparison.md)。使用同一测试集逐波形配对计算，未重新训练。\n',encoding='utf-8')
    print('RESULTS:',out)
    print(selected[selected.group=='all'][['model','observation_time_s','mean_location_error_km','median_location_error_km','p90_location_error_km']].to_string(index=False))
    print(paired[(paired.group=='all')&paired.requested_time_s.isin([20,29.76])].to_string(index=False))
    from summarize_localization import summarize
    summarize()


if __name__=='__main__': main()
