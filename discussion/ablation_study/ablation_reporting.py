"""Four ablation groups, actual-result tables, cautious automatic interpretation."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TASKS=['magnitude','distance','azimuth']
TIMES=[2.88,4.8,9.92,19.84,29.76]
FULL='ELS8K_Full'
PURE='PureGRU_Existing'
SINGLES=['Single_'+task for task in TASKS]
STRATEGIES={'C0_Uniform_Fixed':'C0','C1_Staged_Fixed':'C1','C2_Uniform_Learned':'C2',FULL:'C3'}


def make_plots(root,metrics,resources,loss,cache,write_files=True):
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'plotting'))
    from publication_style import setup,MM
    out=root/'figures';out.mkdir(exist_ok=True)
    names={'zh':['震级MAE','震中距MAE (km)','反方位角MAAE (°)'],
           'en':['Magnitude MAE','Distance MAE (km)','Back-azimuth MAAE (°)']}
    order=resources.set_index('model')
    colors=['#0072B2','#D55E00','#009E73','#CC79A7']
    for language in ['zh','en']:
        font,_,_=setup(language);suffix='CN' if language=='zh' else 'EN'
        last=metrics[np.isclose(metrics.observation_time_s,29.76)] if not metrics.empty else metrics
        def finish(fig,name):
            fig.tight_layout()
            fig.canvas.draw()
            if write_files:
                for ext in ['png','pdf']: fig.savefig(out/f'{name}_{suffix}.{ext}',dpi=600,bbox_inches='tight',pad_inches=.08)
            plt.close(fig)
        if not last.empty:
            arch=last[last.model.isin([FULL,PURE])]
            if not arch.empty:
                fig,axes=plt.subplots(1,5,figsize=(190*MM,65*MM))
                for i,task in enumerate(TASKS):
                    axes[i].bar(arch.model.str.replace('_Existing','',regex=False).str.replace('ELS8K_Full','ELS-8K',regex=False),arch[task+'_mae'],color=colors[:len(arch)])
                    axes[i].set_ylabel(names[language][i],fontproperties=font)
                for ax,column,label in [(axes[3],'total_parameters','Parameters'),(axes[4],'macs_per_update','MACs/update')]:
                    vals=order.loc[arch.model,column]/1000
                    ax.bar(['ELS-8K' if n==FULL else 'PureGRU' for n in arch.model],vals,color=colors[:len(arch)])
                    ax.set_ylabel(label+' (K)')
                for ax in axes: ax.tick_params(axis='x',labelrotation=45,labelsize=7);ax.spines[['top','right']].set_visible(False)
                finish(fig,'FigureA_Architecture')
            if last.model.isin(SINGLES).any() and FULL in last.model.values:
                fig,axes=plt.subplots(1,4,figsize=(190*MM,68*MM))
                for i,task in enumerate(TASKS):
                    part=last[last.model.isin([FULL,'Single_'+task])]
                    axes[i].bar(['Joint' if n==FULL else 'Single' for n in part.model],part[task+'_mae'],color=colors[:len(part)])
                    axes[i].set_ylabel(names[language][i],fontproperties=font)
                if set(SINGLES).issubset(order.index):
                    axes[3].bar(['Joint','3 singles'],[order.loc[FULL,'macs_per_update']/1000,order.loc[SINGLES,'macs_per_update'].sum()/1000],color=colors[:2])
                    axes[3].set_ylabel('MACs/update (K)')
                for ax in axes: ax.spines[['top','right']].set_visible(False)
                finish(fig,'FigureB_Multitask')
        if not loss.empty:
            fig,axes=plt.subplots(1,3,figsize=(190*MM,68*MM))
            for i,task in enumerate(TASKS):
                for j,(name,group) in enumerate(loss.groupby('model',sort=False)):
                    axes[i].plot(group.observation_time_s,group[task+'_mae'],label=STRATEGIES[name],color=colors[j],linestyle=['-','--','-.',':'][j])
                axes[i].set_xlabel('P波后观测时间 (s)' if language=='zh' else 'Time after P arrival (s)',fontproperties=font)
                axes[i].set_ylabel(names[language][i],fontproperties=font);axes[i].legend(fontsize=7)
                axes[i].grid(color='#DDDDDD',linewidth=.4);axes[i].spines[['top','right']].set_visible(False)
            finish(fig,'FigureC_Loss')
        if not cache.empty:
            curve=pd.read_csv(root/'cache/cache_latency_by_update.csv')
            cost=pd.read_csv(root/'cache/cache_cost_by_update.csv')
            fig,axes=plt.subplots(1,3,figsize=(190*MM,68*MM))
            for j,(mode,group) in enumerate(curve.groupby('mode',sort=False)):
                axes[0].plot(group.observation_time_s,group.median_update_ms,label=mode,color=colors[j])
                c=cost[cost['mode']==mode]
                axes[1].plot(c.observation_time_s,c.macs_per_update.cumsum()/1e6,label=mode,color=colors[j])
            axes[0].set_ylabel('Median update latency (ms)');axes[1].set_ylabel('Cumulative MACs (M)')
            for ax in axes[:2]: ax.set_xlabel('Observation time (s)');ax.legend(fontsize=7);ax.grid(color='#DDDDDD',linewidth=.4)
            axes[2].bar(cache['mode'],cache.median_record_total_ms,color=colors[:len(cache)])
            axes[2].set_ylabel(f'{int(cache.updates.iloc[0])}-update model time (ms)')
            for ax in axes: ax.spines[['top','right']].set_visible(False)
            finish(fig,'FigureD_Cache')


def summarize(root,cfg,plots=True):
    plan=pd.read_csv(root/'experiment_plan.csv')
    resources=pd.read_csv(root/'model_resources.csv')
    frames=[];stabilities=[];statuses=[]
    for name in plan.name:
        folder=root/name
        state=json.loads((folder/'state.json').read_text(encoding='utf-8')) if (folder/'state.json').exists() else {'status':'pending'}
        statuses.append(dict(model=name,status=state['status'],error=state.get('error')))
        if state['status']=='complete':
            frames.append(pd.read_csv(folder/'metrics_by_time.csv'));stabilities.append(pd.read_csv(folder/'stability.csv'))
    metrics=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
    stability=pd.concat(stabilities,ignore_index=True) if stabilities else pd.DataFrame()
    stability.to_csv(root/'stability_ablation.csv',index=False)
    keys=metrics[metrics.observation_time_s.apply(lambda t:any(abs(t-x)<1e-8 for x in TIMES))] if not metrics.empty else metrics
    merged=keys.merge(resources,on='model') if not metrics.empty else pd.DataFrame()
    arch=merged[merged.model.isin([FULL,PURE])].copy() if not merged.empty else pd.DataFrame()
    unavailable=pd.DataFrame([dict(model='ParameterMatchedPureGRU',status='unavailable',reason='Existing PureGRU has 5791, baseline 8263; no new PureGRU training per user')])
    pd.concat([arch,unavailable],ignore_index=True).to_csv(root/'architecture_ablation.csv',index=False)
    multi=merged[merged.model.isin([FULL]+SINGLES)].copy() if not merged.empty else pd.DataFrame()
    multi.to_csv(root/'multitask_ablation.csv',index=False)
    loss=metrics[metrics.model.isin(STRATEGIES)].copy() if not metrics.empty else pd.DataFrame()
    if not loss.empty: loss['strategy']=loss.model.map(STRATEGIES)
    loss.to_csv(root/'loss_ablation.csv',index=False)
    selected=loss[loss.observation_time_s.apply(lambda t:any(abs(t-x)<1e-8 for x in TIMES))] if not loss.empty else loss
    selected.to_csv(root/'loss_ablation_key_times.csv',index=False)
    sysrows=[];r=resources.set_index('model')
    if set([FULL]+SINGLES).issubset(r.index):
        columns=['total_parameters','macs_per_update','persistent_state_kib','fp32_weights_kib']
        joint=r.loc[FULL,columns].astype(float);independent=r.loc[SINGLES,columns].sum().astype(float)
        for label,values in [('one_joint_model',joint),('three_independent_single_models',independent)]:
            sysrows.append(dict(system=label,**values.to_dict(),performance_complete=set(SINGLES+[FULL]).issubset(set(metrics.model) if not metrics.empty else set())))
    pd.DataFrame(sysrows).to_csv(root/'multitask_system_costs.csv',index=False)
    cache=pd.DataFrame()
    cstate=root/'cache/state.json'
    if cstate.exists() and json.loads(cstate.read_text(encoding='utf-8')).get('status')=='complete':
        for name in ['cache_consistency.csv','cache_efficiency.csv','cache_cost_by_update.csv','cache_latency_by_update.csv']:
            frame=pd.read_csv(root/'cache'/name);frame.to_csv(root/name,index=False)
        cache=pd.read_csv(root/'cache/cache_efficiency.csv')
    else:
        for name in ['cache_consistency.csv','cache_efficiency.csv']:
            pd.DataFrame(columns=['status','reason']).to_csv(root/name,index=False)
    lines=['# ELS-EEW 8K完整消融实验',
           '全部新模型固定CNN宽度16、GRU两层hidden24，不涉及16K或32K。PureGRU只读复用既有5791参数结构/权重，不重训；8K完整模型复用8263参数最佳权重。',
           '参数匹配PureGRU没有现成权重，遵照补充要求不实施训练，表中明确标记unavailable。当前架构比较不能宣称参数量完全匹配。',
           '单随机种子42，波形记录级统计。测试集不用于选择训练检查点或调整配置；既有测试结果在前期开发中已查看，不声称完整研发流程未接触测试集。',
           '三任务模型沿用至少两项历史验证MAE改善的规则；单任务按自身验证MAE选择，准则不同已明确记录。',
           '统一权重0.85按要求保留。93个输出的四阶段平均权重实际为0.848387，并非精确0.85，记录了约0.19%的尺度差。',
           'MAE单位：震级无量纲、距离/定位km、方位角为最小圆周角差（°）。位置由预测距离和方位角联合计算球面震中，不能由两项平均误差合成。',
           '\n## 运行状态',pd.DataFrame(statuses).to_csv(index=False)]
    if not merged.empty:
        lines+=['\n## 关键时刻精度和资源',merged.to_csv(index=False)]
    if not metrics.empty:
        for time in [2.88,19.84,29.76]:
            part=metrics[np.isclose(metrics.observation_time_s,time)].set_index('model')
            if {FULL,PURE}.issubset(part.index):
                lines.append(f'\n### 架构对比 {time:.2f}s（CNN+GRU减去PureGRU；负值为CNN+GRU误差更小）')
                for task in TASKS+['localization']:
                    column=task+'_mae'
                    delta=part.loc[FULL,column]-part.loc[PURE,column]
                    lines.append(f'- {task}: {delta:+.6f}；'+('本次CNN+GRU较优。' if delta<0 else '本次PureGRU较优或持平，应如实报告。'))
            if FULL in part.index:
                for task,name in zip(TASKS,SINGLES):
                    if name in part.index:
                        delta=part.loc[FULL,task+'_mae']-part.loc[name,task+'_mae']
                        lines.append(f'- {time:.2f}s {task} joint-single MAE={delta:+.6f}；'+('联合较优。' if delta<0 else '独立单任务较优或持平，可能存在任务竞争。'))
                contrasts=[(FULL,'C2_Uniform_Learned','时间加权/学习q'),('C1_Staged_Fixed','C0_Uniform_Fixed','时间加权/固定q'),
                           ('C2_Uniform_Learned','C0_Uniform_Fixed','学习q/统一时间'),(FULL,'C1_Staged_Fixed','学习q/分阶段')]
                for variant,control,label in contrasts:
                    if variant in part.index and control in part.index:
                        delta={task:float(part.loc[variant,task+'_mae']-part.loc[control,task+'_mae']) for task in TASKS}
                        lines.append(f'- {time:.2f}s {label}：处理组减对照组误差 {delta}；逐任务解释，不将不同量纲误差相加。')
    if sysrows:
        lines+=['\n## 三任务系统总成本',pd.DataFrame(sysrows).to_csv(index=False)]
        for col in ['total_parameters','macs_per_update','persistent_state_kib']:
            lines.append(f'- 共享骨干相对三套独立单任务模型的{col}降低 {100*(1-sysrows[0][col]/sysrows[1][col]):.2f}%。')
    if not stability.empty:
        lines+=['\n## 稳定性',stability.to_csv(index=False),
                '绝对变化量按相邻0.32秒输出计算。早/晚窗口的两个端点均须落在指定范围内；同时报告对应预测MAE。常数输出不能被认定为既稳定又准确。']
    if not cache.empty:
        lines+=['\n## 缓存实验',cache.to_csv(index=False),
                '数值差异见cache_consistency.csv，原始四个输出与一次性完整因果前向比较；完整离线前向仅为一致性参考，不是同等在线部署基线。',
                'D0完整前缀重算；D1全部显式缓存；D2仅保留GRU和最近30点原始波形，重算局部CNN。原始波形上下文不是隐藏特征缓存。',
                'MACs只计卷积/矩阵乘加，不计非线性、归一化、残差加法和内存复制。模型状态与输入历史缓冲分别列出。CPU时间为本机batch1、单线程、纯模型推理，不包含文件读取/预处理/绘图。']
        for _,row in cache.iterrows():
            lines.append(f'- {row["mode"]}: 单次更新中位数{row.median_update_ms:.3f}ms、P95={row.p95_update_ms:.3f}ms，'
                         f'整条93次更新累计时间中位数{row.median_record_total_ms:.3f}ms；'
                         f'P95 {"低于" if row.p95_update_ms<320 else "不低于"}320ms更新间隔。仅适用于当前CPU与样本，非嵌入式部署认证。')
    else: lines.append('缓存效率/CPU延迟未完成，不能预设真实加速比或0.32秒周期适用性。')
    lines+=['\n## 未完成部分与结论边界','未完成模型不填入模拟指标，相关研究问题暂不下结论。参数匹配PureGRU不开展，因此该问题保持开放。',
            '结构差异、任务选择准则与单种子随机性均影响解释；不预设完整模型一定最优，不将描述性差异宣称为统计显著。']
    (root/'ablation_summary.md').write_text('\n\n'.join(lines),encoding='utf-8')
    if plots: make_plots(root,metrics,resources,loss,cache)
