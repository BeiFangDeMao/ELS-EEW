"""Actual-data-only summaries, empirical Pareto fronts and preliminary figures."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TASKS={'magnitude_mae':('震级MAE','Magnitude MAE'),
       'distance_mae_km':('震中距MAE (km)','Distance MAE (km)'),
       'azimuth_circular_mae_deg':('反方位角MAAE (°)','Back-azimuth MAAE (°)')}
COSTS=['total_parameters','macs_per_update','persistent_state_kib']


def nondominated(values):
    values=np.asarray(values,dtype=float)
    finite=np.isfinite(values).all(axis=1)
    answer=np.zeros(len(values),dtype=bool)
    for i,row in enumerate(values):
        if finite[i]:
            dominated=finite & (values<=row).all(axis=1) & (values<row).any(axis=1)
            answer[i]=not dominated.any()
    return answer


def pareto_tables(accuracy,resources):
    merged=accuracy.merge(resources,on='model',validate='many_to_one')
    records=[]
    for split in ['validation','test']:
        for time in [3,20]:
            group=merged[(merged.split==split)&(merged.requested_time_s==time)]
            for cost in COSTS:
                for task in list(TASKS)+['joint_three_tasks']:
                    columns=[cost]+(list(TASKS) if task=='joint_three_tasks' else [task])
                    front=nondominated(group[columns].to_numpy())
                    for (_,row),flag in zip(group.iterrows(),front):
                        records.append(dict(split=split,requested_time_s=time,actual_output_time_s=row.actual_output_time_s,
                                            resource=cost,task=task,model=row.model,nondominated=bool(flag),
                                            cost_value=row[cost],error_value=row.get(task,np.nan)))
    return pd.DataFrame(records)


def recommendations(accuracy,resources,cfg):
    rows=[]
    merged=accuracy[accuracy.split=='validation'].merge(resources,on='model')
    for budget in cfg['budgets']:
        for time in [3,20]:
            group=merged[(merged.requested_time_s==time)&(merged.total_parameters<=budget['max_parameters'])&
                         (merged.macs_per_update<=budget['max_macs'])&(merged.persistent_state_kib<=budget['max_state_kib'])]
            if group.empty:
                rows.append(dict(budget=budget['name'],requested_time_s=time,task='all',model=None,reason='No feasible completed candidate'))
                continue
            front=nondominated(group[list(TASKS)].to_numpy())
            for (_,item),flag in zip(group.iterrows(),front):
                if flag: rows.append(dict(budget=budget['name'],requested_time_s=time,task='joint_nondominated',model=item.model,
                                          reason='Validation only; three errors minimized separately; no mixed-unit error sum'))
            for task in TASKS:
                best=group[task].min()
                for _,item in group[group[task]==best].iterrows():
                    rows.append(dict(budget=budget['name'],requested_time_s=time,task=task,model=item.model,
                                     validation_mae=item[task],reason='Minimum validation task error among budget-feasible completed candidates'))
    return pd.DataFrame(rows)


def figures(folder,accuracy,resources,pareto,write_files=True):
    """No data extrapolation, simulated results, or curves through missing models."""
    import sys
    workspace=Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(workspace/'plotting'))
    from publication_style import setup,MM
    output=folder/'figures'
    output.mkdir(parents=True,exist_ok=True)
    rendered=0
    test=accuracy[(accuracy.split=='test')&accuracy.requested_time_s.isin([3,20])].merge(resources,on='model')
    for language in ['zh','en']:
        font,latin,_=setup(language)
        suffix='CN' if language=='zh' else 'EN'
        for number,cost in [(1,'total_parameters'),(2,'macs_per_update')]:
            fig,axes=plt.subplots(1,3,figsize=(190*MM,75*MM))
            fig.subplots_adjust(left=.09,right=.98,bottom=.23,top=.87,wspace=.5)
            for i,(ax,(task,names)) in enumerate(zip(axes,TASKS.items())):
                for time,color,marker in [(3,'#0072B2','o'),(20,'#D55E00','s')]:
                    group=test[test.requested_time_s==time].sort_values(cost)
                    if group.empty: continue
                    x=group[cost].to_numpy()/1000
                    label=f'{time} s ({group.actual_output_time_s.iloc[0]:.2f} s)'
                    ax.plot(x,group[task],color=color,marker=marker,markersize=4,label=label,
                            linestyle='-' if time==3 else '--',linewidth=1.5)
                    if number==2:
                        front=pareto[(pareto.split=='test')&(pareto.requested_time_s==time)&(pareto.task==task)&
                                     (pareto.resource==cost)&pareto.nondominated]
                        flag=group.model.isin(front.model)
                        ax.scatter(x[flag],group.loc[flag,task],s=62,facecolors='none',edgecolors=color,linewidths=.9,zorder=4)
                    for _,row in group.iterrows():
                        ax.annotate(row.model.replace('ELS-',''),(row[cost]/1000,row[task]),xytext=(2,4),textcoords='offset points',fontsize=6.5,fontproperties=latin)
                ax.set_ylabel(names[0 if language=='zh' else 1],fontproperties=font,fontsize=9)
                ax.set_xlabel(('参数量 (K)' if cost=='total_parameters' else '每次更新MACs (K)') if language=='zh' else
                              ('Parameters (K)' if cost=='total_parameters' else 'MACs/update (K)'),fontproperties=font,fontsize=9)
                ax.text(0,1.06,f'({chr(97+i)})',transform=ax.transAxes,fontsize=10,weight='bold')
                ax.grid(color='#DDDDDD',linewidth=.4)
                ax.spines[['top','right']].set_visible(False)
                ax.margins(x=.12,y=.2)
            handles,labels=axes[0].get_legend_handles_labels()
            fig.legend(handles,labels,loc='upper center',ncol=2,frameon=False,fontsize=8)
            fig.canvas.draw()
            rendered+=1
            if write_files:
                for extension in ['pdf','png']: fig.savefig(output/f'Figure{number}_{suffix}.{extension}',dpi=600)
            plt.close(fig)
        fig,axes=plt.subplots(1,3,figsize=(190*MM,72*MM))
        fig.subplots_adjust(left=.09,right=.98,bottom=.23,top=.88,wspace=.55)
        ordered=resources.sort_values('total_parameters')
        for i,(ax,cost) in enumerate(zip(axes,COSTS)):
            values=ordered[cost]/(1000 if i<2 else 1)
            ax.bar(ordered.model.str.replace('ELS-','',regex=False),values,color='#0072B2',width=.65)
            labels_zh=['参数量 (K)','每次更新MACs (K)','持久状态内存 (KiB)']
            labels_en=['Parameters (K)','MACs/update (K)','Persistent state (KiB)']
            ax.set_ylabel((labels_zh if language=='zh' else labels_en)[i],fontproperties=font,fontsize=9)
            ax.set_xlabel('模型规模' if language=='zh' else 'Model scale',fontproperties=font,fontsize=9)
            ax.text(0,1.06,f'({chr(97+i)})',transform=ax.transAxes,fontsize=10,weight='bold')
            ax.set_ylim(0,max(values)*1.15)
            ax.grid(axis='y',color='#DDDDDD',linewidth=.4)
            ax.set_axisbelow(True)
            ax.spines[['top','right']].set_visible(False)
        fig.canvas.draw()
        rendered+=1
        if write_files:
            for extension in ['pdf','png']: fig.savefig(output/f'Figure3_{suffix}.{extension}',dpi=600)
        plt.close(fig)
    return rendered


def summarize(folder,cfg):
    """Rebuild aggregate files from committed per-model outputs after each model."""
    resources=pd.read_csv(folder/'compute_resources.csv')
    states,training,accuracy=[],[],[]
    for model in resources.model:
        model_dir=folder/model
        state=json.loads((model_dir/'state.json').read_text(encoding='utf-8')) if (model_dir/'state.json').exists() else {'status':'pending'}
        states.append(dict(model=model,status=state['status'],error=state.get('error')))
        if (model_dir/'training_summary.json').exists():
            item=json.loads((model_dir/'training_summary.json').read_text(encoding='utf-8'))
            training.append(dict(model=model,status=state['status'],completed_epochs=item['completed_epochs'],best_epoch=item['best_epoch'],reference_retrained=item.get('reference_retrained',True)))
        else:
            training.append(dict(model=model,status=state['status'],completed_epochs=np.nan,best_epoch=np.nan))
        if state['status']=='complete': accuracy.append(pd.read_csv(model_dir/'accuracy_by_time.csv'))
    pd.DataFrame(training).to_csv(folder/'training_summary.csv',index=False)
    complete=sum(item['status']=='complete' for item in states)
    lines=['# ELS-EEW模型规模实验',f'完成 {complete}/5 个候选。未完成模型不产生模拟结果。',
           '单随机种子42。8K参考测试结果在前期研究中已被查看，不将整个开发过程描述为完全未接触测试集。',
           '测试集只报告性能；候选推荐只依据验证集和预设资源预算。不同单位的误差不相加。',
           'MACs是理论矩阵/卷积乘加次数，不是硬件延迟；不包含偏置加法、LayerNorm、GELU、sigmoid/tanh、门控逐元素运算、残差加法和内存复制。',
           '状态为batch=1 FP32的卷积缓存与GRU隐藏状态；不含权重、临时激活、输入缓冲和已在H5中完成的预处理滤波状态。',
           '时间沿用P对齐观测窗：3/5/10/20/29.76 s对应2.88/4.80/9.92/19.84/29.76 s。',
           '原始P到时与裁剪点可能存在0–0.009 s偏移；实际时间仍以原记录对齐信息为准。',
           '\n## 执行状态','\n'.join(f'- {s["model"]}: {s["status"]}'+(f' — {s["error"]}' if s.get('error') else '') for s in states),
           '\n## 资源成本（实际统计）',resources[['model','total_parameters','macs_per_update','persistent_state_kib']].to_csv(index=False)]
    if accuracy:
        accuracy=pd.concat(accuracy,ignore_index=True)
        accuracy.to_csv(folder/'accuracy_by_time.csv',index=False)
        pareto=pareto_tables(accuracy,resources)
        pareto.to_csv(folder/'pareto_results.csv',index=False)
        recommended=recommendations(accuracy,resources,cfg)
        recommended.to_csv(folder/'validation_budget_recommendations.csv',index=False)
        lines+=['\n## 3秒与20秒的三任务误差',accuracy[accuracy.requested_time_s.isin([3,20])].to_csv(index=False),
                '\n## 当前候选集合的经验帕累托前沿（非全局最优证明）']
        for (split,time,cost,task),group in pareto.groupby(['split','requested_time_s','resource','task'],sort=False):
            lines.append(f'- {split}; {time:g}s; {cost}; {task}: '+', '.join(group.loc[group.nondominated,'model']))
        lines+=['\n## 验证集与资源预算下的候选建议',recommended.to_csv(index=False),
                '每个预算分别报告任务最优候选及三任务非支配候选，未强行挑出一个统一主模型。']
        test=accuracy[accuracy.split=='test']
        analysis=[]
        for time in [3,20]:
            group=test[test.requested_time_s==time].set_index('model')
            gains={}
            if {'ELS-2K','ELS-32K'}.issubset(group.index):
                for task in TASKS:
                    small,large=group.loc['ELS-2K',task],group.loc['ELS-32K',task]
                    improvement=100*(small-large)/small if small>0 else np.nan
                    gains[task]=improvement
                    analysis.append(f'- {time:g}s {task}: 2K={small:.6f}, 32K={large:.6f}; 误差相对下降 {improvement:.2f}%。')
                analysis.append(f'- {time:g}s 从2K到32K相对误差下降最多的任务：{max(gains,key=gains.get)}（仅描述性比较）。')
            if len(group)>1:
                order=resources.set_index('model').loc[group.index].sort_values('total_parameters').index
                for task in TASKS:
                    for before,after in zip(order[:-1],order[1:]):
                        if group.loc[after,task]>group.loc[before,task]:
                            analysis.append(f'- {time:g}s {task}: {after}误差高于{before}，增加参数未改善该任务。')
        lines+=['\n## 容量收益、早晚期差异及8K位置',*analysis,
                '对照上述3秒、20秒相对收益，不预设早晚期收益相同；8K的非支配情况见前沿清单。',
                '是否存在比8K更合理的规模，依据验证集预算候选表判断；测试误差较低本身不构成主模型选择规则。']
        figures(folder,accuracy,resources,pareto)
    else:
        pd.DataFrame().to_csv(folder/'accuracy_by_time.csv',index=False)
        pd.DataFrame().to_csv(folder/'pareto_results.csv',index=False)
        lines.append('尚无完成的模型，不能比较精度、敏感任务、早晚期收益或推荐规模。')
    (folder/'summary.md').write_text('\n\n'.join(lines),encoding='utf-8')
