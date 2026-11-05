"""Export six real cases as self-contained CSV/JSON; no inference or training."""
import hashlib
import json
from datetime import datetime
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
import plot_els_event_cases as shared


def write_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def main():
    shared.USE_CORRECTED_NOTO=False
    cases=shared.load_cases()
    saved=pd.read_csv(shared.PRED/'predictions_by_time.csv')
    base=ROOT/'plotting/Plotting_Data_Packages'
    out=base/('Single_Station_6Events_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    out.mkdir(parents=True,exist_ok=False)
    summaries=[]
    common_peak=0.
    for i,c in enumerate(cases):
        name=c['event']['date']+'_'+c['row']['station']
        folder=out/name;folder.mkdir()
        channels=c['channels']
        for channel in channels[1:]:np.testing.assert_array_equal(channel['t'],channels[0]['t'])
        waveform=pd.DataFrame({'time_after_p_s':channels[0]['t']})
        for label,channel in zip('ZNE',channels):
            raw=channel['wave']+channel['baseline_gal']
            waveform['acceleration_'+label+'_gal']=raw
            if label=='Z':common_peak=max(common_peak,float(np.max(np.abs(raw))))
        record=c['row']['record']
        old=saved[saved.record==record].sort_values('chunk').reset_index(drop=True)
        assert len(old)==93
        np.testing.assert_allclose(old.predicted_magnitude,c['pred'][:,0],rtol=1e-7,atol=1e-7)
        p=c['pred'];h=c['row']['header']
        az=np.degrees(np.arctan2(p[:,2],p[:,3]))%360
        table=pd.DataFrame(dict(output_index=np.arange(1,94),nominal_time_after_p_s=c['time'],
            exact_time_after_p_s=old.exact_post_p_s.to_numpy(),predicted_magnitude=p[:,0],
            predicted_epicentral_distance_km=p[:,1],predicted_azimuth_sin=p[:,2],
            predicted_azimuth_cos=p[:,3],predicted_azimuth_deg=az,
            predicted_epicenter_longitude_deg=c['lon'],predicted_epicenter_latitude_deg=c['lat']))
        # CSV doubles preserve all supplied float32 outputs without rounding.
        waveform.to_csv(folder/'waveform.csv',index=False,float_format='%.17g')
        table.to_csv(folder/'predictions.csv',index=False,float_format='%.17g')
        metadata=dict(event_order=i+1,panel_label=chr(97+i),date=c['event']['date'],
            event_name=shared.TITLES[i],event_name_chinese=c['event']['name'],
            station_code=c['row']['station'],record_id=record,
            title=f'({chr(97+i)}) {shared.TITLES[i]} — {c["row"]["station"]}',
            p_pick_seconds_from_record_start=float(c['pick']),p_pick_version='original spreadsheet; no +5.5s correction',
            record_start_header_time=h['record_time'],p_absolute_header_local_time=c['row']['trigger_time'],
            station_longitude_deg=h['station_longitude'],station_latitude_deg=h['station_latitude'],
            true_epicenter_longitude_deg=h['longitude'],true_epicenter_latitude_deg=h['latitude'],
            true_magnitude=float(c['truth'][0]),true_epicentral_distance_km=float(c['truth'][1]),
            true_station_to_epicenter_azimuth_deg=float(c['truth'][2]),
            waveform_physical_unit='gal = cm/s^2',waveform_processing='fixed instrument scale only; no normalization, filtering or mean subtraction',
            waveform_time_range_s=[float(waveform.time_after_p_s.min()),float(waveform.time_after_p_s.max())],
            sampling_rate_hz=float(1/np.median(np.diff(waveform.time_after_p_s))),
            output_count=93,output_step_s=.32,
            crop_p_alignment_offset_s=float(old.exact_post_p_s.iloc[0]-old.nominal_post_p_s.iloc[0]),
            trajectory_method='WGS84 GeographicLib direct: station coordinates + same-output predicted distance and azimuth',
            azimuth_definition='station toward epicenter; clockwise from geographic north',
            source_prediction_file=c['prediction_source'],source_prediction_sha256=shared.sha(c['prediction_source']),
            raw_component_sources=[dict(path=v['path'],sha256=v['sha256']) for v in channels])
        write_json(folder/'metadata.json',metadata)
        summaries.append(dict(event_order=i+1,folder=name,title=metadata['title'],record_id=record,
                              p_pick_seconds_from_record_start=float(c['pick'])))
        check=pd.read_csv(folder/'predictions.csv')
        np.testing.assert_allclose(check.predicted_magnitude,p[:,0],rtol=0,atol=1e-12)
        np.testing.assert_allclose(check.predicted_epicenter_latitude_deg,c['lat'],rtol=0,atol=1e-12)
        np.testing.assert_allclose(pd.read_csv(folder/'waveform.csv').acceleration_Z_gal,
            waveform.acceleration_Z_gal,rtol=1e-14,atol=1e-10)
    limit=float(np.ceil(common_peak/100)*100)
    write_json(out/'events.json',summaries)
    write_json(out/'plot_settings.json',dict(
        current_status='One Noto prototype approved for further visual adjustment; use same real data for all six',
        layout=dict(map_width_height_ratio=1,waveform_width_height_ratio=16/9,equal_width=True,vertical_gap=0,
                    longitude_ticks='top, formatted137.0°E',latitude_ticks='left, formatted37.5°N',
                    longitude_axis_title=False,latitude_axis_title=False,legend='inside map'),
        waveform=dict(component='Z',xlim_s=[-3,29.76],ylim_gal=[-limit,limit],
                      y_ticks_gal=[-limit,0,limit],line_color='#555555',line_width_pt=.4,
                      physical_unit='gal',normalize=False,baseline_subtraction=False),
        magnitude=dict(y_axis='right',ylim=[3,8],y_ticks=[4,6,8],line_width_pt=1.5,
                       true_magnitude_line='red dashed horizontal'),
        trajectory=dict(line_width_pt=.8,station='black upward triangle',true_epicenter='red five-point star',
                        all_points=93,retain_all_points=True,correct_geographic_aspect=True,
                        background='pale land/sea, coastline only; no borders or terrain'),
        time_coloring=dict(colormap='viridis',normalization_s=[0,29.76],
                           line_segment_color='midpoint time of each original segment',
                           scatter_color='output time',shared_by=['trajectory','magnitude']),
        key_outputs=[dict(requested_time_s=3,actual_nominal_time_s=2.88,output_index=9),
                     dict(requested_time_s=10,actual_nominal_time_s=9.92,output_index=31),
                     dict(requested_time_s=20,actual_nominal_time_s=19.84,output_index=62),
                     dict(requested_time_s=29.76,actual_nominal_time_s=29.76,output_index=93)],
        p_line='red dashed vertical at0s',fonts=dict(english='Times New Roman',chinese='SimSun',
            axis_labels_pt=9,ticks_pt=8,legend_pt=8,title_pt=10),export=['vector PDF','600 DPI PNG']))
    map_source=ROOT/'plotting/map_data/ne_10m_land.geojson'
    shutil.copy2(map_source,out/map_source.name)
    shutil.copy2(HERE/'land_background.py',out/'land_background_reference.py')
    write_json(out/'coastline_source.json',dict(source='Natural Earth physical land, public domain',
        url='https://github.com/nvkelso/natural-earth-vector/blob/master/geojson/ne_10m_land.geojson',
        file=map_source.name,sha256=shared.sha(map_source),use='offline GeoJSON; no political boundaries'))
    (out/'README_绘图说明.md').write_text('''# 六事件真实数据绘图交接包

本包自包含：无需模型、H5、PyTorch或原项目。请从events.json按顺序读取六个事件文件夹。
只需CSV/JSON读取工具和绘图工具即可绘图，禁止用模拟数据替换或改写预测结果。

## 文件与字段

- events.json：六事件顺序及对应文件夹。
- 各事件/waveform.csv：time_after_p_s和acceleration_Z/N/E_gal；真实P前3秒至P后29.76秒。
- 各事件/predictions.csv：93行输出；预测震级、震中距、sin/cos、角度、WGS84震中经纬度。
- 各事件/metadata.json：原表P到时、台站和真实震中坐标、真实震级/距离/方位和来源记录。
- plot_settings.json：用户已确定的版式、统一幅值范围、渐变色、线宽和关键时刻。
- ne_10m_land.geojson：真实陆地轮廓，可直接用于离线陆地/海洋背景，不含国界。
- land_background_reference.py：可参考的Matplotlib多边形绘制函数；使用时将SOURCE改为本包GeoJSON路径。

## 已确定的版式

每个事件一个贴合矩形：上图1:1、下图16:9、两图等宽且无间隙。
上图为单台轨迹，黑三角标台站，红五角星标真实震中；顶部经度刻度写137.0°E，左侧纬度写37.5°N，不写Longitude/Latitude轴标题。
上图中放图例及时间渐变色条。轨迹线0.8pt，震级线1.5pt，二者统一viridis渐变，时间归一化0–29.76秒。
下图为原始Z分量加速度和右轴震级；全部六事件波形纵轴统一-1200到1200gal，不能逐图归一化。
P到时0秒画红竖虚线；真实震级画红横虚线。
关键输出点为2.88/9.92/19.84/29.76秒，分别是第9/31/62/93行（从1开始）。
先画一个能登样图确认，再扩展六事件。不要添加双台或三台结果。

## 时间与P到时的关键说明

最新指示是使用原表P到时。能登ISK003为13.52秒，不使用曾试验的19.02秒，也不使用排除ISK003后的ISK007首台版本。
预测只使用与原P到时匹配的原推理结果。waveform.csv为相对原标注P时刻的真实采样时间。
nominal_time_after_p_s是原评估网格0.32、0.64…29.76秒；exact_time_after_p_s另包含裁剪对齐偏移，通常0–0.009秒。
复刻当前样图使用nominal时间；若要求严格绝对时间对齐，则统一使用exact时间（含轨迹颜色与标注）。不要混淆，也不能插值虚构精确3.00秒预测。
“3秒点”当前指不晚于3秒的2.88秒输出，应明确标注实际值。

## 波形、轨迹的真实性

波形仅原仪器固定系数换算为gal，无归一化、无额外滤波、无去均值；前3秒是真实原始记录，不补零。
预测文件为原有物理范围裁剪后的真实模型输出，没有平滑或修改；每个坐标来自同一行的距离与方位。
WGS84坐标已计算并正反算核对，直接使用longitude/latitude字段即可，画图不必再次计算。
角度定义是台站指向震中、正北0°、顺时针。不要把该角再加180°。
真实经纬度和真实震级仅作为参考标记，不用于修正预测。
所有明显偏离的预测点必须保留，不能因美观删点。
原项目以6371km球面距离评估；本包轨迹为用户要求的WGS84显示坐标，差异不代表预测数值改变。
实际保留样本的端点可能与-3/29.76秒相差不足一个原采样间隔，不伪造恰好落在边界的样本。

## 原项目的调用关系（仅追溯时需要）

最新单事件样图：discussion/real_events/plot_one_case_prototype.py。
共享数据读取：plot_els_event_cases.py -> 设置USE_CORRECTED_NOTO=False -> load_cases()。
原文本读取：plot_first_waveforms.py -> read_wave(path)，返回原记录时间/真实gal波形/采样率。
背景：land_background.py -> add_land_sea(ax)。
原始索引：discussion/real_events/results/20261022_230231_957410/first_stations.json。
原预测：discussion/real_events/results/compactgru_first_20261022_231155_186077/processed_predictions.npy，shape=(6,93,4)，顺序为震级、震中距km、sin方位、cos方位。
真实标签：同目录truths.npy，shape=(6,3)，顺序为震级、距离km、方位度。

完整性校验：package_manifest.json记录全部文件SHA256。仅需本包就能绘图，绝对源路径可忽略。
''',encoding='utf-8')
    files={str(p.relative_to(out)):shared.sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
    write_json(out/'package_manifest.json',dict(version=1,events=6,files=files,script=str(Path(__file__)),
               data_checks='six events; 93 outputs per case; waveforms and coordinate CSV roundtrip verified; raw file hashes verified',
               excluded='corrected picks and all multistation predictions'))
    archive=shutil.make_archive(str(out),'zip',root_dir=out.parent,base_dir=out.name)
    print('PACKAGE:',out,flush=True);print('ZIP:',archive,flush=True)


if __name__=='__main__':main()
