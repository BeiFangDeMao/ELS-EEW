# -*- coding: utf-8 -*-
"""Readable interpretation from saved localization results; no new inference."""
from pathlib import Path
import json
import pandas as pd

HERE=Path(__file__).resolve().parent


def table(headers,rows):
    return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+
                     ['|'+'|'.join(map(str,row))+'|' for row in rows])


def summarize():
    out=Path(json.loads((HERE/'latest_result.json').read_text(encoding='utf-8'))['result_directory'])
    meta=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
    run=Path(meta['source_experiment'])
    metrics=pd.read_csv(out/'localization_key_times.csv')
    pairs=pd.read_csv(out/'paired_comparison.csv')
    order=['ELS-8K','ELS-16K','ELS-32K']
    final=metrics[(metrics.group=='all')&(metrics.requested_time_s==29.76)].set_index('model')
    mean='mean_location_error_km'
    rows=[]
    for time in [3,5,10,20,29.76]:
        frame=metrics[(metrics.group=='all')&(metrics.requested_time_s==time)].set_index('model')
        rows.append([f'{frame.observation_time_s.iloc[0]:.2f}']+[f'{frame.loc[m,mean]:.2f}' for m in order])
    lines=['# 8K、16K、32K定位精度：补充分析',
           '使用同一独立测试集的51772条波形、同一批次、同一93点输出时间轴重新推理；未重新训练或微调。',
           '定位误差是预测震中与真实震中的球面距离。使用原距离/方位角后处理和R=6371 km球面正向定位，并复用Experiment01计算函数。',
           '## 平均定位误差随观测时间变化（km）',table(['实际观测时间/s','8K','16K','32K'],rows),
           '## 最终29.76秒的误差分布与达标率',
           table(['模型','均值/km','中位数/km','P90/km','误差≤10km','误差≤20km','误差≤30km'],
                 [[m,f'{final.loc[m,mean]:.2f}',f'{final.loc[m,"median_location_error_km"]:.2f}',
                   f'{final.loc[m,"p90_location_error_km"]:.2f}']+
                  [f'{100*final.loc[m,f"within_{d}km"]:.1f}%' for d in [10,20,30]] for m in order]),
           '## 改善幅度与逐波形比较']
    for _,row in pairs[(pairs.group=='all')&(pairs.requested_time_s==29.76)].iterrows():
        lines.append(f'- {row.smaller}→{row.larger}：平均误差减少{row.mean_reduction_km:.2f} km（{row.relative_reduction_pct:.1f}%）；'
                     f'{100*row.larger_model_better_fraction:.1f}%波形的定位误差变小。')
    early=pairs[(pairs.group=='all')&(pairs.smaller=='ELS-8K')&(pairs.larger=='ELS-32K')].set_index('requested_time_s')
    lines+=['## 如何理解',
            f'1. 32K相对8K在2.88秒时平均定位误差下降{early.loc[3,"mean_reduction_km"]:.2f} km（{early.loc[3,"relative_reduction_pct"]:.1f}%），'
            f'到19.84秒下降{early.loc[20,"mean_reduction_km"]:.2f} km（{early.loc[20,"relative_reduction_pct"]:.1f}%）。容量收益在后期更明显。',
            f'2. 最终32K不仅降低均值，也将P90从{final.loc["ELS-8K","p90_location_error_km"]:.2f} km降到'
            f'{final.loc["ELS-32K","p90_location_error_km"]:.2f} km，说明大误差尾部也改善。定位≤20 km比例从'
            f'{100*final.loc["ELS-8K","within_20km"]:.1f}%增至{100*final.loc["ELS-32K","within_20km"]:.1f}%，'
            f'增加{100*(final.loc["ELS-32K","within_20km"]-final.loc["ELS-8K","within_20km"]):.1f}个百分点。',
            '3. 这与规模实验中距离和方位角两项均改善的趋势一致，但联合定位误差必须逐波形计算，不能直接用两个平均误差合成。',
            '4. 32K每次更新MACs约为8K的4.05倍。其定位收益有实际幅度，但仍有长尾，不能据此称为稳定的10 km精度定位。',
            '## 真实震中距分组：最终平均定位误差（km）']
    groups=[('D_le_10','≤10 km'),('D_le_50','≤50 km'),('D_50_100','[50,100) km'),('D_100_200','[100,200) km')]
    rows=[]
    for group,label in groups:
        frame=metrics[(metrics.group==group)&(metrics.requested_time_s==29.76)].set_index('model')
        rows.append([label,str(int(frame.records.iloc[0]))]+[f'{frame.loc[m,mean]:.2f}' for m in order])
    lines+=[table(['真实震中距','记录数','8K','16K','32K'],rows),
            '≤10和≤50是累计范围，有重叠；[50,100)、[100,200)是区间。不要将这些行相加。精确边界按已保存距离标签判断。',
            '≤10 km组的16K与32K均值很接近，且32K并非所有分位数、达标率均更好；不应声称大模型对每个距离组、每条波形都更优。',
            '## 论文中可采用的表述',
            '增大模型容量同时改善了距离与方向估计，并转化为更低的联合定位误差。在29.76 s观测窗口下，8K、16K和32K模型的平均定位误差分别为'
            f'{final.loc["ELS-8K",mean]:.2f}、{final.loc["ELS-16K",mean]:.2f}和{final.loc["ELS-32K",mean]:.2f} km。'
            f'32K相对8K降低约{early.loc[29.76,"relative_reduction_pct"]:.1f}%，同时每次流式更新的理论MACs约增加至4.05倍，体现了定位精度与计算开销之间的权衡。',
            '## 评价边界与文件',
            '单随机种子、波形记录级描述性比较；不作事件级或统计显著性结论。测试结果仅报告性能，不用于替代既定验证集预算选型规则。',
            f'全部93个时刻及距离分组：`{out / "localization_by_time.csv"}`；逐波形配对比较：`{out / "paired_comparison.csv"}`。',
            '原始预测、定位误差数组、真值坐标、记录键及权重SHA256均已保存，可直接复算其他容差或绘图，不必重新推理。',
            '距离与角度误差已逐项核对原规模实验表，最大允许差异1e-3；全部权重在推理前后SHA256不变。']
    text='\n\n'.join(lines)+'\n'
    (out/'interpretation.md').write_text(text,encoding='utf-8')
    (run/'localization_comparison.md').write_text(text,encoding='utf-8')
    print(run/'localization_comparison.md')


if __name__=='__main__': summarize()
