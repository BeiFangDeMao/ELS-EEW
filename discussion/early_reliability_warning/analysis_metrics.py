"""Regression reliability and irrevocable current-output threshold decisions."""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,precision_recall_curve,auc,average_precision_score

from config import *
from inference import dump


def divide(a,b):
    a,b = np.broadcast_arrays(np.asarray(a,dtype=float),np.asarray(b,dtype=float))
    return np.divide(a,b,out=np.full(a.shape,np.nan),where=b!=0)


class RecordCI:
    """95% Wilson binomial-proportion intervals, with waveforms as units."""
    def __init__(self,n):
        self.n = n

    def totals(self,values):
        values = np.asarray(values,dtype=np.float64)
        if values.ndim==1: values = values[:,None]
        return values.sum(axis=0)

    def interval(self,numerator,denominator):
        z = 1.959963984540054
        proportion = divide(numerator,denominator)
        denominator = np.asarray(denominator,dtype=float)
        scale = 1+divide(z*z,denominator)
        center = (proportion+divide(z*z,2*denominator))/scale
        half = z*np.sqrt(divide(proportion*(1-proportion),denominator)+divide(z*z,4*denominator**2))/scale
        return np.clip(center-half,0,1),np.clip(center+half,0,1),np.isfinite(proportion)

    def rates(self,values):
        return self.interval(self.totals(values),self.totals(np.ones(self.n)))


def metrics(tp,fp,tn,fn):
    result = dict(TP=tp,FP=fp,TN=tn,FN=fn,
        accuracy=divide(tp+tn,tp+fp+tn+fn),precision=divide(tp,tp+fp),recall=divide(tp,tp+fn),
        f1=divide(2*tp,2*tp+fp+fn),specificity=divide(tn,tn+fp),
        false_positive_rate=divide(fp,fp+tn),false_negative_rate=divide(fn,fn+tp))
    result['balanced_accuracy'] = (result['recall']+result['specificity'])/2
    result['mcc'] = divide(tp*tn-fp*fn,np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
    return result


def strategy_specs():
    return [dict(name=f'M_ge_{m:.1f}',kind='magnitude',m=m,d=None) for m in (4.,5.,6.)]+[
        dict(name=f'D_le_{d}',kind='distance',m=None,d=d) for d in (50,100)]+[
        dict(name=name,kind='joint',m=m,d=d) for name,m,d in (
            ('joint_A_M4_D50',4,50),('joint_B_M4_D100',4,100),
            ('joint_C_M6_D50',6,50),('joint_D_M6_D100',6,100))]


def condition(spec,mag,dis):
    value = np.ones_like(mag,dtype=bool)
    if spec['m'] is not None: value &= mag>=spec['m']
    if spec['d'] is not None: value &= dis<=spec['d']
    return value


def first_trigger(current):
    cumulative = np.maximum.accumulate(current,axis=1)
    exists = current.any(axis=1)
    first = np.where(exists,current.argmax(axis=1),-1)
    return cumulative,first


def validate_decisions():
    # Alarm remains on after current predictions fall below threshold.
    cumulative,first = first_trigger(np.array([[False,True,False,True],[False,False,False,False]]))
    assert cumulative.tolist()==[[False,True,True,True],[False,False,False,False]]
    assert first.tolist()==[1,-1]
    # Joint conditions reached at different times do not generate an alarm.
    spec = dict(m=4,d=50)
    joint = condition(spec,np.array([[5.,3.]]),np.array([[100.,10.]]))
    assert not joint.any()
    m = metrics(np.array([0.]),np.array([0.]),np.array([2.]),np.array([1.]))
    assert np.isnan(m['precision'][0]) and m['recall'][0]==0 and np.isnan(m['mcc'][0])
    assert REPORT_TIMES[0]==1 and np.floor(1/.32)*.32==.96
    lo,hi,valid = RecordCI(10).interval(np.array([0.,10.]),np.array([10.,10.]))
    assert np.allclose(lo,[0,.7224672],atol=1e-6) and np.allclose(hi,[.2775328,1],atol=1e-6)
    assert valid.all()


def regression(out,meta,pred,truth,location):
    t = pred.shape[1]
    times = (np.arange(t)+1)*CHUNK/RATE
    angle = np.rad2deg(np.arctan2(pred[:,:,2].astype(np.float64),pred[:,:,3].astype(np.float64)))%360
    errors = dict(magnitude=abs(pred[:,:,0].astype(np.float64)-truth[:,0:1]),
                  distance_km=abs(pred[:,:,1].astype(np.float64)-truth[:,1:2]),
                  azimuth_deg=abs((angle-truth[:,2:3]+180)%360-180),location_km=location)
    ci = RecordCI(len(meta))
    regression_rows,tolerance_rows = [],[]
    groups = [('all',np.ones(len(meta),dtype=bool))]+[
        (f'distance_{lo}_{hi}',(truth[:,1]>=lo)&(truth[:,1]<hi)) for lo,hi in ((0,10),(10,50),(50,100),(100,200))]
    for group,mask in groups:
        selected_meta = meta.loc[mask]
        if not mask.any(): continue
        estimator = ci if group=='all' else RecordCI(len(selected_meta))
        for task,array in errors.items():
            if group!='all' and task!='location_km': continue
            e = np.asarray(array[mask],dtype=np.float64)
            med,p90 = np.percentile(e,[50,90],axis=0,method='linear')
            for j in range(t):
                regression_rows.append(dict(scope='all_records',group=group,task=task,chunk=j+1,
                    observation_time_s=times[j],records=int(mask.sum()),
                    exact_post_p_time_min_s=times[j]+float(selected_meta.p_alignment_offset_s.min()),
                    exact_post_p_time_max_s=times[j]+float(selected_meta.p_alignment_offset_s.max()),
                    mae=float(e[:,j].mean()),rmse=float(np.sqrt(np.mean(e[:,j]**2))),
                    median_ae=med[j],p90_ae=p90[j],
                    change_mae_vs_final=float(e[:,j].mean()-e[:,-1].mean()),
                    change_rmse_vs_final=float(np.sqrt(np.mean(e[:,j]**2))-np.sqrt(np.mean(e[:,-1]**2))),
                    change_median_ae_vs_final=med[j]-med[-1],change_p90_ae_vs_final=p90[j]-p90[-1]))
            for tolerance in TOLERANCES[task]:
                success = e<=tolerance
                rates = success.mean(axis=0)
                low,high,valid = estimator.rates(success)
                for j in range(t):
                    tolerance_rows.append(dict(scope='all_records',group=group,task=task,tolerance=tolerance,
                        chunk=j+1,observation_time_s=times[j],records=int(mask.sum()),successes=int(success[:,j].sum()),
                        exact_post_p_time_min_s=times[j]+float(selected_meta.p_alignment_offset_s.min()),
                        exact_post_p_time_max_s=times[j]+float(selected_meta.p_alignment_offset_s.max()),
                        rate=rates[j],ci95_low=low[j],ci95_high=high[j],ci_defined=bool(valid[j]),
                        ci_method='95% Wilson binomial proportion interval, waveform records as units',
                        change_rate_vs_final=rates[j]-rates[-1]))
    r,tol = pd.DataFrame(regression_rows),pd.DataFrame(tolerance_rows)
    r.to_csv(out/'regression_metrics.csv',index=False,encoding='utf-8-sig')
    tol.to_csv(out/'tolerance_rates.csv',index=False,encoding='utf-8-sig')
    crossings = []
    for (group,task,tolerance),df in tol.groupby(['group','task','tolerance'],sort=False):
        for target in (.8,.9,.95):
            indices = np.flatnonzero(df.rate.to_numpy()>=target)
            crossings.append(dict(group=group,task=task,tolerance=tolerance,target_rate=target,
                first_observation_time_s=float(df.observation_time_s.iloc[indices[0]]) if len(indices) else np.nan,
                reached=bool(len(indices)),note='Descriptive first crossing; may later fall below target, not a deployment rule'))
    pd.DataFrame(crossings).to_csv(out/'overall_rate_first_crossings.csv',index=False,encoding='utf-8-sig')
    return r,tol


def warning_scope(out,scope,indices,meta,pred,truth):
    m = meta.iloc[indices].reset_index(drop=True)
    p = np.asarray(pred[indices],dtype=np.float64)
    y = truth[indices]
    n,t = p.shape[:2]
    times = (np.arange(t)+1)*CHUNK/RATE
    ci = RecordCI(len(m))
    rows,alert_frames,supports,oscillation,auc_rows = [],[],[],[],[]
    offsets = m.p_alignment_offset_s.to_numpy(dtype=float)
    for spec in strategy_specs():
        actual = condition(spec,y[:,0],y[:,1])
        current = condition(spec,p[:,:,0],p[:,:,1])
        cumulative,first = first_trigger(current)
        assert np.all(np.diff(cumulative.astype(int),axis=1)>=0)
        pos = int(actual.sum())
        tp_mask = cumulative&actual[:,None]
        fp_mask = cumulative&~actual[:,None]
        tp,fp = tp_mask.sum(axis=0).astype(float),fp_mask.sum(axis=0).astype(float)
        fn,tn = pos-tp,n-pos-fp
        scores = metrics(tp,fp,tn,fn)
        boot_tp,boot_fp = ci.totals(tp_mask),ci.totals(fp_mask)
        boot_positives = ci.totals(actual)
        prec_lo,prec_hi,prec_valid = ci.interval(boot_tp,boot_tp+boot_fp)
        rec_lo,rec_hi,rec_valid = ci.interval(boot_tp,boot_positives)
        alarm_times = np.where(first>=0,(first+1)*CHUNK/RATE+offsets,np.nan)
        for j,time in enumerate(times):
            correct_times = alarm_times[actual&(first>=0)&(first<=j)]
            row = dict(scope=scope,strategy=spec['name'],kind=spec['kind'],magnitude_threshold=spec['m'],
                distance_threshold_km=spec['d'],chunk=j+1,observation_time_s=time,records=n,
                exact_post_p_time_min_s=time+float(offsets.min()),exact_post_p_time_max_s=time+float(offsets.max()),
                cumulative_state_available_by_s=time+float(offsets.max()),
                positive_records=pos,positive_fraction=pos/n,negative_records=n-pos,small_positive_support=pos<20,
                **{k:float(v[j]) for k,v in scores.items()},
                precision_ci95_low=prec_lo[j],precision_ci95_high=prec_hi[j],precision_ci_defined=bool(prec_valid[j]),
                recall_ci95_low=rec_lo[j],recall_ci95_high=rec_hi[j],recall_ci_defined=bool(rec_valid[j]),
                cumulative_correct_alert_rate=divide(tp[j],pos).item(),cumulative_false_alert_count=int(fp[j]),
                true_positive_first_alert_time_mean_s=float(np.mean(correct_times)) if len(correct_times) else np.nan,
                true_positive_first_alert_time_median_s=float(np.median(correct_times)) if len(correct_times) else np.nan,
                true_positive_first_alert_time_p90_s=float(np.percentile(correct_times,90)) if len(correct_times) else np.nan,
                final_unalerted_target_fraction=divide(fn[-1],pos).item(),
                change_precision_vs_final=float(scores['precision'][j]-scores['precision'][-1]),
                change_recall_vs_final=float(scores['recall'][j]-scores['recall'][-1]),
                change_f1_vs_final=float(scores['f1'][j]-scores['f1'][-1]))
            rows.append(row)
            for metric_name,series in scores.items():
                if metric_name not in ('TP','FP','TN','FN'):
                    row['change_'+metric_name+'_vs_final'] = float(series[j]-series[-1])
        frame = m[['record_id','p_arrival_absolute']].copy()
        frame['scope'],frame['strategy'] = scope,spec['name']
        frame['actual_target'] = actual
        frame['first_alert_chunk'] = np.where(first>=0,first+1,np.nan)
        frame['first_alert_time_after_p_s'] = alarm_times
        frame['first_alert_absolute_time'] = m.p_arrival_absolute+pd.to_timedelta(alarm_times,unit='s')
        frame['first_alert_time_nominal_s'] = np.where(first>=0,(first+1)*CHUNK/RATE,np.nan)
        frame['ever_alerted'] = first>=0
        frame['final_outcome'] = np.where(actual,np.where(first>=0,'TP','FN'),np.where(first>=0,'FP','TN'))
        frame['raw_condition_dropped_after_first'] = ((~current)&cumulative).any(axis=1)
        ups = np.sum(np.diff(np.pad(current.astype(int),((0,0),(1,0))),axis=1)==1,axis=1)
        frame['raw_condition_upcrossing_count'] = ups
        alert_frames.append(frame)
        alerted = first>=0
        oscillation.append(dict(scope=scope,strategy=spec['name'],records=n,ever_alerted=int(alerted.sum()),
            dropped_after_first_count=int(frame.raw_condition_dropped_after_first.sum()),
            dropped_after_first_fraction_among_alerted=divide(frame.raw_condition_dropped_after_first.sum(),alerted.sum()).item(),
            repeated_upcrossing_count=int((ups>1).sum()),
            repeated_upcrossing_fraction_among_alerted=divide((ups>1).sum(),alerted.sum()).item()))
        supports.append(dict(scope=scope,strategy=spec['name'],records=n,positive_records=pos,negative_records=n-pos,
            positive_fraction=pos/n,small_positive_support=pos<20,
            reason='Fewer than 20 positive waveform records; descriptive heuristic, not a statistical standard' if pos<20 else ''))
        if spec['kind']=='magnitude':
            for j,time in enumerate(times):
                score = p[:,j,0]
                if pos==0 or pos==n:
                    roc,pr,ap = np.nan,np.nan,np.nan
                    reason = 'Only one true class present'
                else:
                    roc = roc_auc_score(actual,score)
                    precision,recall,_ = precision_recall_curve(actual,score)
                    pr = auc(recall,precision)
                    ap = average_precision_score(actual,score)
                    reason = ''
                auc_rows.append(dict(scope=scope,strategy=spec['name'],chunk=j+1,observation_time_s=time,
                    roc_auc=roc,pr_auc_trapezoidal=pr,average_precision=ap,positive_records=pos,reason=reason,
                    note='Current clipped magnitude score, not cumulative alert state'))
    pd.DataFrame(oscillation).to_csv(out/f'oscillation_{scope}.csv',index=False,encoding='utf-8-sig')
    return pd.DataFrame(rows),pd.concat(alert_frames,ignore_index=True),pd.DataFrame(supports),pd.DataFrame(auc_rows)


def analyze(out):
    validate_decisions()
    meta = pd.read_parquet(out/'predictions/records.parquet')
    pred = np.load(out/'predictions/processed_predictions.npy',mmap_mode='r')
    truth = np.load(out/'predictions/truths.npy')
    location = np.load(out/'predictions/location_errors_km.npy',mmap_mode='r')
    print('Computing waveform reliability and Wilson CIs',flush=True)
    regression_metrics,tolerance_rates = regression(out,meta,pred,truth,location)
    results = []
    for scope,mask in [('record_level',np.ones(len(meta),dtype=bool))]:
        print('Computing warning metrics: '+scope,flush=True)
        results.append(warning_scope(out,scope,np.flatnonzero(mask),meta,pred,truth))
    warning = pd.concat([r[0] for r in results],ignore_index=True)
    warning.to_csv(out/'warning_metrics.csv',index=False,encoding='utf-8-sig')
    alerts = pd.concat([r[1] for r in results],ignore_index=True)
    alerts.to_csv(out/'alert_times.csv',index=False,encoding='utf-8-sig')
    alerts.to_parquet(out/'alert_times.parquet',index=False)
    supports = pd.concat([r[2] for r in results],ignore_index=True)
    supports.to_csv(out/'class_support.csv',index=False,encoding='utf-8-sig')
    aucs = pd.concat([r[3] for r in results],ignore_index=True)
    aucs.to_csv(out/'magnitude_score_auc.csv',index=False,encoding='utf-8-sig')
    mapping = []
    for target in REPORT_TIMES:
        actual_times = (np.arange(pred.shape[1])+1)*CHUNK/RATE
        offsets = meta.p_alignment_offset_s.to_numpy(dtype=float)
        indices = np.flatnonzero(actual_times+offsets.max()<=target+1e-10)
        j = int(indices[-1])
        mapping.append(dict(requested_time_s=target,chunk=j+1,actual_nominal_time_s=actual_times[j],
            min_actual_time_after_p_s=actual_times[j]+offsets.min(),max_actual_time_after_p_s=actual_times[j]+offsets.max(),
            valid_records=len(meta)))
    pd.DataFrame(mapping).to_csv(out/'report_time_mapping.csv',index=False,encoding='utf-8-sig')
    chosen = [r['chunk'] for r in mapping]+[pred.shape[1]]
    warning.loc[warning.chunk.isin(chosen)].to_csv(out/'warning_metrics_key_times.csv',index=False,encoding='utf-8-sig')
    tolerance_rates.loc[tolerance_rates.chunk.isin(chosen)].to_csv(out/'tolerance_rates_key_times.csv',index=False,encoding='utf-8-sig')
    write_summary(out,meta,regression_metrics,tolerance_rates,warning,supports,mapping,aucs)
    dump(out/'statistics_manifest.json',dict(analysis_unit='waveform_record',event_or_station_ids_computed=False,
        interval='95% Wilson binomial-proportion interval; records as analysis units; no event/station grouping',
        interval_limitation='Binomial record-level intervals do not account for possible dependence among waveforms; not event-level generalization intervals.',
        undefined='CSV NaN represented by empty cells; JSON nonfinite values null. No zero imputation.',
        undefined_precision='No cumulative alerts',undefined_recall='No actual positives',
        undefined_specificity='No actual negatives',undefined_balanced_accuracy='Either class absent',
        undefined_mcc='Any MCC denominator factor zero',undefined_auc='Only one actual class',
        primary_warning_scope='record_level',small_support_heuristic='Fewer than 20 positive waveform records',completed=True))


def fmt(value,digits=3):
    return 'NaN' if pd.isna(value) else f'{value:.{digits}f}'


def write_summary(out,meta,r,tol,w,support,mapping,aucs):
    audit = json.loads((out/'metadata_audit.json').read_text(encoding='utf-8'))
    lines = ['# 实验1：CompactGRU早期可靠性与不可撤回阈值决策','',
        '本次重新加载指定原始最佳权重并完整推理，未训练、未微调、未修改权重或架构。',
        f"测试波形记录 {len(meta):,} 条，93个输出，0.32–29.76 s；全部统计以单条波形为单位。",'',
        '## 评价范围','',
        *['- '+x for x in audit['limitations']],
        '- 按最新要求，不计算事件或台站ID，不归组、不选首台，每条波形独立运行不可撤回报警状态。',
        '- 达标率、Precision和Recall的95%CI使用记录级Wilson二项比例区间；不代表事件级区间，未考虑不同记录可能存在的相关性。',
        '- 正负样本数量见class_support.csv；少于20条正波形标记为样本不足，20仅为描述性提示阈值。',
        '- 阈值只模拟震源参数决策，不是强震动或用户地点的完整预警系统；真实标签仅用于事后评分。','',
        '## 时间映射','',
        '|请求时刻/s|实际名义时刻/s|有效记录|','|---:|---:|---:|']
    for row in mapping: lines.append(f"|{row['requested_time_s']}|{row['actual_nominal_time_s']:.2f}|{row['valid_records']}|")
    lines += ['', '精确P后时间还包含每条记录的裁剪对齐偏移，min/max见report_time_mapping.csv；映射不晚于请求时刻。','',
        '逐chunk表中的observation_time_s为对齐窗的名义时刻，真实P后可用时间区间见exact_post_p_time_min/max_s；',
        '累计报警状态全部可用的截止时刻见cumulative_state_available_by_s，不能把名义时刻解释为无对齐偏移的精确统一截止时刻。','',
        '## 全记录参数可靠性','',
        '|时刻/s|震级MAE|震级AE≤0.3|距离MAE/km|角MAAE/°|定位均值/km|定位≤10km|定位≤20km|',
        '|---:|---:|---:|---:|---:|---:|---:|---:|']
    chunks = [x['chunk'] for x in mapping]+[93]
    for chunk in chunks:
        rr = r[(r.group=='all')&(r.chunk==chunk)].set_index('task')
        tt = tol[(tol.group=='all')&(tol.chunk==chunk)]
        def rate(task,value): return tt.loc[(tt.task==task)&(tt.tolerance==value),'rate'].iloc[0]*100
        lines.append(f"|{chunk*.32:.2f}|{rr.loc['magnitude','mae']:.3f}|{rate('magnitude',.3):.1f}%|{rr.loc['distance_km','mae']:.2f}|{rr.loc['azimuth_deg','mae']:.2f}|{rr.loc['location_km','mae']:.2f}|{rate('location_km',10):.1f}%|{rate('location_km',20):.1f}%|")
    lines += ['', '所有误差容差达标率保留逐时刻波动，没有累计化或强制单调处理。',
        '距离分组为[0,10)、[10,50)、[50,100)、[100,200)，互斥；只对真实距离分组，不按预测筛选。','',
        '## 震级阈值：不可撤回累计报警（记录级）','',
        '|阈值|真实正记录|时刻/s|Precision|Recall|F1|FP|FN|TP首次报警中位/s|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name in ('M_ge_4.0','M_ge_5.0','M_ge_6.0'):
        selected = w[(w.scope=='record_level')&(w.strategy==name)&w.chunk.isin(chunks)]
        for _,row in selected.iterrows():
            lines.append(f"|{name}|{int(row.positive_records)}|{row.observation_time_s:.2f}|{fmt(row.precision)}|{fmt(row.recall)}|{fmt(row.f1)}|{int(row.FP)}|{int(row.FN)}|{fmt(row.true_positive_first_alert_time_median_s,2)}|")
    lines += ['', 'TP报警时间仅统计截至该时刻已正确报警的目标记录，是条件统计，必须结合Recall理解。','',
        '## 其他策略最终结果（记录级）','',
        '|策略|正记录|Precision|Recall|F1|FPR|最终漏报比例|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for _,row in w[(w.scope=='record_level')&(w.chunk==93)&(w.kind!='magnitude')].iterrows():
        lines.append(f"|{row.strategy}|{int(row.positive_records)}|{fmt(row.precision)}|{fmt(row.recall)}|{fmt(row.f1)}|{fmt(row.false_positive_rate)}|{fmt(row.final_unalerted_target_fraction)}|")
    lines += ['', '## 样本支持与初步讨论','']
    for _,row in support[(support.scope=='record_level')].iterrows():
        lines.append(f"- {row.strategy}：正波形 {int(row.positive_records)}，负波形 {int(row.negative_records)}，正样本比例 {row.positive_fraction:.3%}。"+(' 正样本不足，避免仅凭Accuracy下结论。' if row.small_positive_support else ''))
    rate03 = tol[(tol.group=='all')&(tol.task=='magnitude')&(tol.tolerance==.3)]
    crossing = rate03.loc[rate03.rate>=.8,'observation_time_s']
    lines += ['', '### 本次结果中的具体观察','',
        f"- 震级AE<=0.3的记录比例首次达到80%的名义时刻为 {crossing.iloc[0]:.2f} s。" if len(crossing) else '- 震级AE<=0.3的记录比例未达到80%。',
        f"- 最终震级AE<=0.3达标率为 {rate03.rate.iloc[-1]:.2%}；90%和95%的目标"+('均未达到。' if rate03.rate.max()<.9 else '请见overall_rate_first_crossings.csv。')]
    for name in ('M_ge_4.0','M_ge_5.0','M_ge_6.0'):
        final = w[(w.scope=='record_level')&(w.strategy==name)&(w.chunk==93)].iloc[0]
        roc = aucs[(aucs.scope=='record_level')&(aucs.strategy==name)&(aucs.chunk==93)].roc_auc.iloc[0]
        lines.append(f"- {name}最终累计Precision={final.precision:.2%}、Recall={final.recall:.2%}、FPR={final.false_positive_rate:.2%}；当前震级得分ROC-AUC={roc:.4f}。当前得分区分能力与累计不可撤回误报不是同一指标。")
    oscillation = pd.read_csv(out/'oscillation_record_level.csv')
    for name in ('M_ge_5.0','M_ge_6.0'):
        row = oscillation.loc[oscillation.strategy==name].iloc[0]
        lines.append(f"- {name}曾报警的波形中，{row.dropped_after_first_fraction_among_alerted:.2%}之后至少一次不再满足当前阈值；这些报警仍不可撤回。该比例不是误报率，真实正波形也可能回落。")
    rare = support[(support.scope=='record_level')&(support.positive_fraction<.02)]
    for _,row in rare.iterrows():
        lines.append(f"- {row.strategy}正类占比仅{row.positive_fraction:.2%}（{int(row.positive_records)}条）；Accuracy容易被大量负类主导，宜同时看Precision、Recall和正类样本数。")
    lines += ['', '- 单次提前越阈会永久保留为报警；即使后面预测回落，FP也不会撤销。累计Recall非递减，但Precision和F1无需单调。',
        '- 原始逐时刻越阈振荡见oscillation_*.csv；首次报警后回落不意味着系统重新变成未报警。',
        '- 当前得分ROC-AUC、PR-AUC与不可撤回累计报警是不同问题；PR-AUC采用梯形积分，另保存Average Precision。',
        '- 固定时间相对最终时刻的变化与首次达到80/90/95%容差达标率仅为测试集描述，未用来调部署策略。',
        '- 无连续噪声与非地震记录，因此FP是已筛选地震记录的参数阈值误报，不是地震事件检测误报。',
        '- 后续可讨论早期震级可靠性、一次误触发的不可逆影响和不同阈值的正负波形比例。记录数不等于独立地震事件数，不计算事件数量。不得据此直接声称端到端用户预警提前量。','',
        '## 输出文件','',
        '- predictions/predictions.parquet：完整逐波形逐时间预测、未裁剪输出、裁剪输出、记录键/坐标/时间、预测震中及定位误差。',
        '- predictions/records.parquet：原始记录元数据，不含生成的事件/台站ID或首台标记。',
        '- predictions/*.npy：原始/处理预测、真值、定位误差，供快速重新统计。',
        '- regression_metrics.csv、tolerance_rates.csv：全记录及互斥距离组回归统计、达标率和CI。',
        '- warning_metrics.csv、alert_times.csv/.parquet：每条波形的累计报警统计、首次报警时间。',
        '- class_support.csv、magnitude_score_auc.csv、overall_rate_first_crossings.csv、oscillation_*.csv。',
        '- minimal_validation.json、inference_manifest.json、metadata_audit.json、statistics_manifest.json记录检查和定义。',
        '- 所有CSV使用UTF-8 BOM；未定义数值为空单元格（NaN），不填0；JSON中非有限数为null。']
    (out/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
