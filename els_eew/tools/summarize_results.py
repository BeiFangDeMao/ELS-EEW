"""Export campaign comparisons without importing torch or rerunning inference."""
import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.excel_results import (
    CHUNK_HEADERS, add_table, chunk_rows, csv_prediction_rows, export_model_excel,
    require_excel_dependency, save_workbook,
)

MAE_FIELDS = tuple(f'{task}_{scope}_mae' for task in ('mag', 'dis', 'azi_deg')
                   for scope in ('overall', 'final'))
MAE_HEADERS = ["震级总体MAE(M)", "震级最终MAE(M)", "距离总体MAE(km)", "距离最终MAE(km)",
               "方位角总体MAAE(deg)", "方位角最终MAAE(deg)"]


def _reexport_model(folder, report, run, settings):
    # Existing Excel has unrounded inference values: do not replace it using a rounded CSV.
    if list(folder.glob('results_*.xlsx')):
        return
    history_path = folder / 'training_history_full.json'
    history = json.loads(history_path.read_text(encoding='utf-8')) if history_path.exists() else {}
    csv_paths = list(folder.glob('test_table_*.csv'))
    metadata = dict(settings, **report.get('metadata', {}))
    metadata.update(model_id=run['model_id'], seed=run['seed'],
                    prediction_source='已保存的CSV，保留CSV数值精度',
                    history_source='training_history_full.json' if history else '旧结果未保存完整JSON训练历史')
    rows = csv_prediction_rows(csv_paths[0], len(report['by_chunk'])) if len(csv_paths) == 1 else ()
    if len(csv_paths) != 1:
        metadata['prediction_status'] = '未找到唯一预测CSV，逐样本预测为空'
    export_model_excel(folder / f"results_{run['model_id']}.xlsx", report, metadata, history, rows)


def summarize(campaign, reexport_missing=True, export_excel=True):
    campaign = Path(campaign).resolve()
    manifest = json.loads((campaign / 'experiment_manifest.json').read_text(encoding='utf-8'))
    rows, blocks, statuses = [], [], []
    seen = set()
    for run in manifest['runs']:
        identity = (run['model_id'], run['seed'])
        if identity in seen:
            raise ValueError(f"重复模型/随机种子：{identity}")
        seen.add(identity)
        status = run['status']
        paths = list(Path(run['output_dir']).glob('checkpoint_*/test_metrics_by_chunk.json'))
        test_status = 'not_completed'
        if status == 'completed':
            if manifest.get('settings', {}).get('skip_final_test'):
                test_status = 'test_skipped'
            elif len(paths) != 1:
                test_status = 'missing_metrics' if not paths else 'multiple_metric_files'
            else:
                path = paths[0]
                report = json.loads(path.read_text(encoding='utf-8'))
                test_status = 'completed'
                metadata = report.get('metadata', {})
                row = dict(model_id=run['model_id'], seed=run['seed'],
                           num_parameters=metadata.get('num_parameters'),
                           best_epoch=metadata.get('best_epoch'),
                           sequence_length=report.get('sequence_length'),
                           test_records=metadata.get('test_records'), result_dir=str(path.parent))
                for task in ('mag', 'dis', 'azi_deg'):
                    row[f'{task}_overall_mae'] = report['overall'][task].get('mae')
                    row[f'{task}_final_mae'] = report['last_chunk']['metrics'][task].get('mae')
                rows.append(row)
                blocks.extend([run['model_id'], run['seed']] + values for values in chunk_rows(report))
                if export_excel and reexport_missing:
                    _reexport_model(path.parent, report, run, manifest.get('settings', {}))
        statuses.append([run['model_id'], run['seed'], status, test_status,
                         run.get('exit_code'), run['output_dir']])
    for model in manifest.get('models', []):
        for seed in manifest.get('seeds', []):
            if (model, seed) not in seen:
                statuses.append([model, seed, 'not_started', 'not_completed', None, None])
    fieldnames = ['model_id', 'seed', 'num_parameters', 'best_epoch', 'sequence_length',
                  'test_records'] + list(MAE_FIELDS) + ['result_dir']
    with (campaign / 'comparison_summary.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # 训练流程仅写入小体积CSV；Excel仍可由手动导出入口生成。
    if not export_excel:
        return rows

    workbook = require_excel_dependency().Workbook(write_only=True)
    add_table(workbook, '模型对比',
              ['模型', '随机种子', '参数量(含任务损失权重)', '最佳Epoch', '输入采样点',
               '测试记录数'] + MAE_HEADERS + ['结果目录'],
              ([row.get(field) for field in fieldnames] for row in rows))
    groups = defaultdict(list)
    for row in rows:
        groups[row['model_id']].append(row)
    seed_stats = []
    for model, group in groups.items():
        for field, label in zip(MAE_FIELDS, MAE_HEADERS):
            values = [row[field] for row in group if isinstance(row[field], (int, float))
                      and math.isfinite(row[field])]
            seed_stats.append([model, label, len(values),
                               statistics.mean(values) if values else None,
                               statistics.stdev(values) if len(values) > 1 else None])
    add_table(workbook, '多种子统计', ['模型', '指标', '有效种子数', '均值', '样本标准差'], seed_stats)
    add_table(workbook, '逐输出点指标', ['模型', '随机种子'] + CHUNK_HEADERS, blocks)
    add_table(workbook, '运行状态', ['模型', '随机种子', '训练状态', '测试状态', '退出码', '输出目录'], statuses)
    settings = dict(manifest.get('settings', {}))
    settings['总体指标口径'] = '测试集全部输出点平均；final为最后输出点；所有角度指标采用圆周MAAE'
    settings['种子统计口径'] = '仅成功测试且指标有限的种子；标准差为ddof=1，单个种子不计算'
    settings['空白数值含义'] = '未保存、未计算、不适用或非有限值，不代表零'
    add_table(workbook, '实验配置', ['字段', '值'], settings.items())
    destination = save_workbook(workbook, campaign / 'comparison_summary.xlsx')
    print(f"Excel汇总已保存：{destination}", flush=True)
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('campaign', type=Path)
    summarize(parser.parse_args().campaign)
