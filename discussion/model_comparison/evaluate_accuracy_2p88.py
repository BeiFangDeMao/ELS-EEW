"""Unified eight-row accuracy table from saved predictions; BazNet components replay only."""
import csv,hashlib,importlib.util,json,sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
from tqdm import tqdm

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
ORIGINAL=HERE/'training_results/20261022_212803_440932'
BAZ=HERE/'additional_baselines/training_results/20261023_234041_581709'
ELS=ROOT/'discussion/early_reliability_warning/results/20261018_180137_947801_record_only'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()

def dump(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def read_prediction(path,keys,truth):
    frame=pd.read_csv(path,dtype={'key':str})
    if frame['key'].duplicated().any():raise ValueError(f'Duplicate IDs: {path}')
    actual=set(frame.key);expected=set(keys)
    audit=dict(file=str(path),records=len(frame),intersection=len(actual&expected),missing=len(expected-actual),extra=len(actual-expected))
    if actual!=expected:raise ValueError(f'Cannot silently subset: {audit}')
    aligned=frame.set_index('key').loc[keys]
    stored=aligned.truth.to_numpy(dtype=float)
    pred=aligned.prediction.to_numpy(dtype=float)
    assert np.isfinite(stored).all() and np.isfinite(pred).all()
    # CSV labels were float32 training targets; canonical evaluation uses the common float64 labels.
    difference=float(np.max(abs(stored-truth)))
    np.testing.assert_allclose(stored,truth,rtol=0,atol=2e-5)
    audit['max_truth_float32_rounding_difference']=difference
    return pred,audit

def infer_baz_components(folder,keys,sources):
    identity=json.loads((BAZ/'identity.json').read_text())
    adapter=HERE/'additional_baselines/models.py'
    assert sha(adapter)==identity['code_sha256'][str(adapter.resolve())],'BazNet implementation changed'
    spec=importlib.util.spec_from_file_location('_accuracy_baz_adapter',adapter)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    sys.path.insert(0,str(HERE))
    from comparison_dataset import ComparisonDataset
    dataset=ComparisonDataset(Path(identity['dataset']),'test','baz')
    np.testing.assert_array_equal(dataset.keys,keys)
    checkpoint=BAZ/'baz_network/best.pth'
    sources[str(checkpoint)]=sha(checkpoint);sources[str(adapter)]=sha(adapter)
    model=module.create_model('baz_network')
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model.load_state_dict(state['model'],strict=True)
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(4)
    model.to(device).eval()
    outputs=[];ids=[]
    # Same trained batch size, preprocessing, feature construction and model implementation.
    with torch.inference_mode():
        for wave,target,batch_keys in tqdm(DataLoader(dataset,batch_size=identity['config']['batch_size'],shuffle=False,num_workers=0),desc='BazNet component replay'):
            cosine,sine=model(wave.to(device))
            components=torch.cat([sine,cosine],dim=1).cpu().numpy()
            assert np.isfinite(components).all()
            outputs.append(components);ids.extend(batch_keys)
    dataset.close();np.testing.assert_array_equal(ids,keys)
    components=np.concatenate(outputs).astype(np.float64)
    assert components.shape==(51772,2)
    np.save(folder/'BazNet_raw_sin_cos.npy',components)
    dump(folder/'BazNet_replay.json',dict(device=str(device),eval=True,inference_mode=True,records=len(ids),
        samples=288,time_s=2.88,checkpoint_epoch_zero_based=state['epoch'],
        checkpoint_sha256=sha(checkpoint),adapter_matches_training_identity=True,
        reason='Saved test CSV contains angles only. Recover original unnormalized components for the established pooled sin/cos R2.'))
    return components

def main():
    folder=HERE/'accuracy_2p88_results'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    folder.mkdir(parents=True);(folder/'predictions').mkdir()
    sources={};audits=[];comparisons=[]
    def track(path):sources[str(path)]=sha(path);return path
    manifest=json.loads(track(ELS/'inference_manifest.json').read_text())
    assert manifest['test_records']==51772 and manifest['num_outputs']==93 and manifest['checkpoint_epoch']==258
    keys=pd.read_parquet(track(ELS/'predictions/records.parquet'),columns=['record_id']).record_id.to_numpy(dtype=str)
    truth=np.load(track(ELS/'predictions/truths.npy'))
    all_pred=np.load(track(ELS/'predictions/processed_predictions.npy'),mmap_mode='r')
    assert all_pred.shape==(51772,93,4) and truth.shape==(51772,7) and len(set(keys))==51772
    index=8
    assert (index+1)*32==288 and abs(manifest['nominal_output_times_s'][index]-2.88)<1e-12
    with np.load(track(ROOT/'plotting/Fig01_Data_Distribution/labels.npz'),allow_pickle=False) as c:
        mask=c['splits']=='test'
        np.testing.assert_array_equal(keys,c['keys'][mask])
        np.testing.assert_array_equal(truth[:,:3],c['labels'][mask])
    for run in [ORIGINAL,BAZ]:
        ident=json.loads(track(run/'identity.json').read_text())
        assert ident['cohort_sha256']==sha(Path(ident['dataset'])/'cohort.npz')
        with np.load(track(Path(ident['dataset'])/'cohort.npz'),allow_pickle=False) as c:
            mask=c['splits']=='test'
            np.testing.assert_array_equal(c['keys'][mask],keys)
            np.testing.assert_array_equal(c['labels'][mask],truth[:,:3])
    prediction=np.asarray(all_pred[:,index,:],dtype=np.float64)
    theta=np.deg2rad(truth[:,2]);true_components=np.column_stack([np.sin(theta),np.cos(theta)])
    baz_components=infer_baz_components(folder,keys,sources)
    baz_saved,baz_audit=read_prediction(track(BAZ/'baz_network/test_predictions.csv'),keys,truth[:,2])
    restored=np.rad2deg(np.arctan2(baz_components[:,0],baz_components[:,1]))%360
    max_angle_difference=float(np.max(abs((restored-baz_saved+180)%360-180)))
    assert max_angle_difference<.001, f'BazNet replay mismatch {max_angle_difference} degrees'
    baz_audit['raw_component_replay_max_circular_angle_difference_deg']=max_angle_difference
    audits.append(baz_audit)
    configurations=[('Magnitude','ELS-EEW',0,None),('Magnitude','SeisT-S',0,'seist_s_emg'),
        ('Magnitude','MagNet',0,'magnet'),('Distance','ELS-EEW',1,None),('Distance','SeisT-S',1,'seist_s_dis'),
        ('Back-azimuth','ELS-EEW',2,None),('Back-azimuth','SeisT-S',2,'seist_s_baz'),('Back-azimuth','BazNet',2,'baz_network')]
    historical=json.loads(track(HERE/'paper_table/comparison_values.json').read_text())
    original=json.loads(track(ORIGINAL/'summary.json').read_text())
    els_original=json.loads(track(ROOT/'els_eew/outputs/experiment_20261012_223631_490021/CompactGRU_seed42/checkpoint_CompactGRU_20261012_235806/test_metrics_by_chunk.json').read_text())
    rows=[];counts=[]
    for task,model,column,name in configurations:
        y=truth[:,column]
        if model=='ELS-EEW':
            p=prediction[:,column] if column!=2 else np.rad2deg(np.arctan2(prediction[:,2],prediction[:,3]))%360
            components=prediction[:,2:4] if column==2 else None
            audits.append(dict(model=model,task=task,records=len(keys),intersection=len(keys),missing=0,extra=0))
        elif model=='BazNet':p=baz_saved;components=baz_components
        else:
            p,audit=read_prediction(track(ORIGINAL/name/'test_predictions.csv'),keys,y)
            audit.update(model=model,task=task);audits.append(audit)
            components=np.column_stack([np.sin(np.deg2rad(p)),np.cos(np.deg2rad(p))]) if column==2 else None
        error=((p-y+180)%360)-180 if column==2 else p-y
        ae=abs(error)
        if column==2:
            assert ae.max()<=180
            target=true_components;estimate=components
        else:target=y;estimate=p
        sst=float(np.square(target-target.mean(axis=0)).sum())
        score=float(1-np.square(estimate-target).sum()/sst) if sst>0 else None
        check=r2_score(target,estimate,multioutput='variance_weighted',force_finite=False)
        assert score is None or abs(score-check)<1e-12
        row=dict(Task=task,Model=model,MAE=float(ae.mean()),MSE=float(np.square(error).mean()),MedAE=float(np.median(ae)),R2=score)
        rows.append(row);counts.append(dict(Task=task,Model=model,records=len(keys),common_intersection=51772))
        detailed=pd.DataFrame(dict(sample_id=keys,task=task,model=model,nominal_time_s=2.88,true_value=y,prediction=p,signed_error=error,absolute_error=ae))
        if column==2:
            detailed['true_sin']=true_components[:,0];detailed['true_cos']=true_components[:,1]
            detailed['pred_sin']=components[:,0];detailed['pred_cos']=components[:,1]
        detailed.to_csv(folder/'predictions'/f'{model}_{task}.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
        if model=='BazNet':old_mae=json.loads(track(BAZ/'baz_network/completed.json').read_text())['mae'];old_r2=None
        elif model=='ELS-EEW':
            metric=['mag','dis','azi_deg'][column]
            old_mae=els_original['by_chunk'][index]['metrics'][metric]['mae']
            old_r2=historical['ELS-EEW'][['magnitude_r2','distance_r2','baz_r2'][column]]
        else:old_mae=original[name]['mae'];old_r2=historical[name]['r2']
        mae_difference=row['MAE']-old_mae
        r2_difference=score-old_r2 if old_r2 is not None else None
        assert abs(mae_difference)<5e-5
        if r2_difference is not None:assert abs(r2_difference)<5e-6
        comparisons.append(dict(model=model,task=task,old_mae=old_mae,new_mae=row['MAE'],mae_difference=mae_difference,
            old_r2=old_r2,new_r2=score,r2_difference=r2_difference,
            explanation='Common float64 labels versus saved float32 training targets/reduction; no changed predictions. BazNet R2 newly calculated from raw components.' if model!='ELS-EEW' else 'Same stored outputs; old float32 aggregate versus float64 statistics.'))
    for path,expected in sources.items():assert sha(path)==expected,f'Source changed: {path}'
    payload=dict(headers=['Task','Model','MAE','MSE','MedAE','R2'],rows=rows,records=51772,common_intersection=51772,
        time_s=2.88,output_chunk=9,last_input_sample_zero_based=287,counts=counts,sample_audits=audits,
        historical_checks=comparisons,source_sha256=sources,
        baz_r2='1 - sum((pred_sin-true_sin)^2+(pred_cos-true_cos)^2) / sum((true_sin-mean(true_sin))^2+(true_cos-mean(true_cos))^2). No predicted vector normalization. SeisT scalar angle is encoded as unit sin/cos; BazNet uses original raw outputs reordered from (cos,sin) to (sin,cos).',
        baz_replay_max_angle_difference_deg=max_angle_difference,analysis_unit='waveform_record')
    dump(folder/'evaluation.json',payload)
    lines=['# 四模型2.88 s统一预测精度统计','',
        '每项均包含51,772条波形记录，8项任务的样本ID集合与交集均为51,772，无缺失、重复或静默删样本。',
        'ELS采用第9次输出（数组索引8、输入末点零基索引287）；其余模型使用现有288点数据及合法特征构造，无填充、无新增滤波、无读取第289点及之后的波形。',
        '时间沿用论文名义P对齐时长2.88 s；原数据相对标注P到时存在0–0.009 s对齐偏差。因此不是对所有记录均精确重采样到人工标注P后2.880 s的另一套评价。','',
        '|Task|Model|MAE / MAAE|MSE|MedAE|R²|','|---|---|---:|---:|---:|---:|']
    for r in rows:lines.append('|{}|{}|{:.4f}|{:.4f}|{:.4f}|{}|'.format(r['Task'],r['Model'],r['MAE'],r['MSE'],r['MedAE'],'NaN' if r['R2'] is None else f"{r['R2']:.4f}"))
    lines+=['','## 方法和单位','',
        '震级/距离：e=预测−真值，MAE=mean(abs(e))，MSE=mean(e²)，MedAE=median(abs(e))，R²=1−SSE/SST。震级无量纲，距离MAE/MedAE单位km、MSE单位km²。',
        '反方位角：e=((预测角−真实角+180)%360)−180，MAE在报告中称MAAE，单位°；MSE单位°²，MedAE单位°。',
        '全部角度定义均为从台站指向震中、从北顺时针。BazNet标签训练为(cosθ,sinθ)，恢复为atan2(sinθ,cosθ)；协方差/特征向量仅基于288点。使用与训练身份哈希一致的本地适配模型，不再修改特征构造。',
        '反方位角R²沿用已确定的sin/cos联合定义：'+payload['baz_r2'],
        '上述同一公式保留模型原本的输出幅值：ELS与BazNet原始分量不单位化，SeisT仅有角度故编码为单位向量。这意味着该R²也反映分量幅值误差，不能仅解读为角度精度；未对0–360°角度直接计算线性R²。','',
        '## 预测复用与核对','',
        'ELS、三个SeisT任务、MagNet和BazNet的角度误差指标均读取已有逐记录预测。仅BazNet因缺少保存的原始双分量，用best.pth、model.eval()、torch.inference_mode()补做一次完整推理以计算联合R²；逐记录恢复角度与旧CSV的最大圆周差为 {:.9g}°。'.format(max_angle_difference),
        '所有模型重新计算的MAE与历史结果差异均小于5e-5，ELS/SeisT/MagNet的R²差异均小于5e-6。差异来自统一使用原始float64真值，而旧CSV写入的是float32训练标签，以及旧float32汇总舍入；所有详细差值保存在evaluation.json。BazNet没有旧联合R²可核对。',
        '所有旧文件哈希在运行前后核对不变。不重新训练，不调参，不进行任何资源或速度测量。',
        '主CSV包含8行，保留完整浮点精度；本报告展示4位小数。predictions目录分任务保存8份逐记录CSV，包含真实值、预测值、样本ID、有符号误差和绝对误差；角度任务还保存R²使用的原始sin/cos。',
        'SeisT三个任务各自对应独立权重；MagNet仅震级，BazNet仅方位角。不把同一事件的多个台站记录声称为独立事件。']
    (folder/'summary.md').write_text('\n\n'.join(lines).replace('|\n\n|','|\n|'),encoding='utf-8')
    dump(folder.parent/'latest_run.json',dict(directory=str(folder)))
    print(pd.DataFrame(rows).to_string(index=False,float_format=lambda x:f'{x:.8f}'))
    print('RESULT_DIRECTORY='+str(folder),flush=True)

if __name__=='__main__':main()
