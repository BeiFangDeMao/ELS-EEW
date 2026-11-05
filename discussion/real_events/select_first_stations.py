"""Match six K-NET events, rank annotated arrivals, save JSON and raw triplets."""
import argparse
from collections import defaultdict
from datetime import datetime, timedelta
import hashlib
import json
import math
from pathlib import Path
import re
import tarfile

import pandas as pd

HERE = Path(__file__).resolve().parent
EVENTS = [('20240101','能登半岛',7.6,16),('20240808','日向滩',7.1,31),
          ('20240417','丰后水道',6.6,39),('20230505','能登地区',6.5,12),
          ('20240315','福岛县近海',5.8,50),('20230511','千叶县南部',5.2,40)]


def header(stream):
    lines = [stream.readline().decode('ascii').strip() for _ in range(17)]
    def value(prefix):
        return next(line[len(prefix):].strip() for line in lines if line.startswith(prefix))
    return dict(origin_time=value('Origin Time'), latitude=float(value('Lat.')),
                longitude=float(value('Long.')), depth_km=float(value('Depth. (km)')),
                magnitude=float(value('Mag.')), station=value('Station Code'),
                record_time=value('Record Time'), sampling_rate_hz=float(re.search(
                    r'[\d.]+',value('Sampling Freq(Hz)')).group()),
                duration_s=float(value('Duration Time(s)')),
                station_latitude=float(value('Station Lat.')),
                station_longitude=float(value('Station Long.')))


def load_picks(path):
    picks = {}
    sheets = []
    for name, frame in pd.read_excel(path,sheet_name=None,header=None).items():
        rows = 0
        for row in frame.itertuples(index=False,name=None):
            if len(row)<2 or pd.isna(row[0]) or pd.isna(row[1]): continue
            key = Path(str(row[0]).strip()).name
            key = re.sub(r'\.(EW|NS|UD)$','',key,flags=re.I)
            if not re.fullmatch(r'[A-Za-z0-9]{6}\d{10}',key):
                raise ValueError(f'无法识别记录名：{name}: {row[0]}')
            p = float(row[1])
            if not math.isfinite(p) or p<0: raise ValueError(f'无效P到时：{key}')
            if key in picks and abs(picks[key]-p)>1e-9:
                raise ValueError(f'重复记录P到时冲突：{key}')
            picks[key] = p; rows += 1
        sheets.append(dict(sheet=name,rows=rows))
    return picks,sheets


def inspect_archive(path):
    with tarfile.open(path,'r|gz') as archive:
        for m in archive:
            if m.isfile() and Path(m.name).suffix.upper() in ('.UD','.NS','.EW'):
                return header(archive.extractfile(m))
    raise ValueError(f'包内没有K-NET三分量文件：{path}')


def rank_event(path,picks,out):
    groups = defaultdict(dict)
    with tarfile.open(path,'r|gz') as archive:
        for member in archive:
            suffix = Path(member.name).suffix.upper()
            if member.isfile() and suffix in ('.EW','.NS','.UD'):
                key = Path(member.name).stem
                if suffix in groups[key]: raise ValueError(f'重复分量：{key}{suffix}')
                groups[key][suffix] = dict(member=member.name,header=header(archive.extractfile(member)))
    eligible,excluded = [],[]
    for key, components in groups.items():
        reason = None
        if set(components)!=set(('.EW','.NS','.UD')): reason='incomplete_triplet'
        elif key not in picks: reason='missing_p_pick'
        if reason:
            excluded.append(dict(record=key,reason=reason)); continue
        h = components['.UD']['header']; p = picks[key]
        if any(c['header']!=h for c in components.values()):
            excluded.append(dict(record=key,reason='inconsistent_component_headers')); continue
        if p>=h['duration_s']:
            excluded.append(dict(record=key,reason='p_outside_record')); continue
        start = datetime.strptime(h['record_time'],'%Y/%m/%d %H:%M:%S')
        trigger = start+timedelta(seconds=p)
        eligible.append(dict(record=key,station=h['station'],p_arrival_s=p,
            trigger_time=trigger.isoformat(timespec='microseconds'),header=h,
            members={k:v['member'] for k,v in components.items()}))
    eligible.sort(key=lambda r:(r['trigger_time'],r['station'],r['record']))
    selected=[]; seen=set()
    for row in eligible:
        if row['station'] in seen: continue
        seen.add(row['station']); selected.append(dict(row,rank=len(selected)+1))
        if len(selected)==3: break
    # No extractall: save only selected regular files under controlled filenames.
    wanted={member:(row,ext) for row in selected for ext,member in row['members'].items()}
    if wanted:
        folder=out/path.name.split('.')[0]; folder.mkdir()
        with tarfile.open(path,'r|gz') as archive:
            for m in archive:
                if m.isfile() and m.name in wanted:
                    row,ext=wanted[m.name]; content=archive.extractfile(m).read()
                    target=folder/(row['record']+ext)
                    target.write_bytes(content)
                    row.setdefault('saved_files',{})[ext]=dict(path=str(target),sha256=hashlib.sha256(content).hexdigest())
    return dict(archive=str(path),total_records=len(groups),eligible_records=len(eligible),
                selected=selected,ranked_records=eligible,excluded=excluded,
                status='ranked' if selected else 'no_eligible_records')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--picks',type=Path,default=HERE/'P波到时-knet24-候哥筛选.xlsx')
    args=parser.parse_args()
    picks,sheets=load_picks(args.picks)
    out=HERE/'results'/datetime.now().strftime('%Y%m%d_%H%M%S_%f'); out.mkdir(parents=True)
    results=[]
    for date,name,mag,depth in EVENTS:
        candidates=[]
        for path in sorted((HERE/'K').glob(f'{date[:4]}/{date[:6]}/{date}*.knt.tar.gz')):
            h=inspect_archive(path)
            if abs(h['magnitude']-mag)<=0.11:
                candidates.append((path,h))
        entry=dict(date=date,name=name,requested_magnitude=mag,requested_depth_km=depth,
                   candidates=[dict(archive=str(p),header=h) for p,h in candidates])
        if len(candidates)==1:
            entry.update(rank_event(candidates[0][0],picks,out))
        else: entry['status']='ambiguous_event' if candidates else 'event_not_found'
        results.append(entry)
        print(name,entry['status'],[(r['station'],r['trigger_time']) for r in entry.get('selected',[])],flush=True)
    report=dict(picks_file=str(args.picks.resolve()),sheets=sheets,unique_picks=len(picks),
                timezone='Asia/Tokyo (header local time, no conversion)',
                trigger_rule='Record Time + annotated P seconds, exactly as requested; no additional timing correction',
                scope='earliest among complete triplets with valid annotations, not guaranteed network-wide first',
                tie_rule='equal trigger times ordered by station then record; ties do not imply physical ordering',
                waveform_qc='complete triplet and consistent headers; raw numeric waveform quality not assessed',events=results)
    (out/'first_stations.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(out/'first_stations.json',flush=True)


if __name__=='__main__': main()
