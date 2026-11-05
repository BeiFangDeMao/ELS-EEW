"""Infer exactly the six selected first records from the original causal test H5."""
import hashlib
import json
from datetime import datetime
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'discussion/early_reliability_warning'))
from inference import load_model, clamp
from config import SOURCE, WEIGHTS, CACHE


def main():
    selection=HERE/'results/20261022_230231_957410/first_stations.json'
    report=json.loads(selection.read_text(encoding='utf-8'))
    selected=[(e,e['selected'][0]) for e in report['events']]
    if len(selected)!=6: raise ValueError('需要六条第一台记录')
    with np.load(CACHE,allow_pickle=False) as cache:
        test_keys=set(cache['keys'][cache['splits']=='test'])
    waves=[]; truths=[]; offsets=[]
    with h5py.File(SOURCE,'r') as source:
        for event,row in selected:
            key=row['record']
            if key not in test_keys: raise ValueError(f'不在原独立测试集合：{key}')
            ds=source[key]
            wave=ds[:].astype(np.float32).T
            if wave.shape!=(3,2976) or not np.isfinite(wave).all(): raise ValueError(key)
            waves.append(wave)
            truths.append([float(ds.attrs[k]) for k in ('mag','dis','azi')])
            offsets.append(float(ds.attrs.get('p_alignment_offset_s',0)))
    model,device,checkpoint=load_model()
    device=torch.device('cpu')
    model=model.to(device)
    x=torch.from_numpy(np.stack(waves)).to(device)
    with torch.inference_mode():
        full=model(x)[0]
        state=None; parts=[]
        for chunk in x.split(32,dim=-1):
            pred,state=model.stream_step(chunk,state); parts.append(pred)
        raw=torch.cat(parts,dim=1)
        torch.testing.assert_close(raw,full,rtol=2e-4,atol=2e-5)
        raw=raw.cpu().numpy()
        processed=clamp(raw)
    out=HERE/'results'/('compactgru_first_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True)
    np.save(out/'raw_predictions.npy',raw)
    np.save(out/'processed_predictions.npy',processed)
    np.save(out/'truths.npy',np.array(truths))
    rows=[]; summaries=[]
    for i,(event,record) in enumerate(selected):
        mag,dist,azi=truths[i]
        for j,p in enumerate(processed[i]):
            angle=float(np.degrees(np.arctan2(p[2],p[3]))%360)
            delta=abs(angle-azi)%360
            rows.append(dict(event_date=event['date'],record=record['record'],station=record['station'],
                chunk=j+1,nominal_post_p_s=(j+1)*.32,
                exact_post_p_s=(j+1)*.32+offsets[i],
                true_magnitude=mag,predicted_magnitude=float(p[0]),magnitude_ae=abs(float(p[0])-mag),
                true_distance_km=dist,predicted_distance_km=float(p[1]),distance_ae_km=abs(float(p[1])-dist),
                true_azimuth_deg=azi,predicted_azimuth_deg=angle,azimuth_ae_deg=min(delta,360-delta)))
        summaries.append(dict(event_name=event['name'],record=record['record'],station=record['station'],
                              at_2_88_s=rows[-93+8],at_29_76_s=rows[-1]))
    pd.DataFrame(rows).to_csv(out/'predictions_by_time.csv',index=False,encoding='utf-8-sig')
    manifest=dict(model='CompactGRU',checkpoint=str(WEIGHTS),checkpoint_epoch=int(checkpoint['epoch']),
        checkpoint_sha256=hashlib.sha256(WEIGHTS.read_bytes()).hexdigest(),source_h5=str(SOURCE),
        selection=str(selection),records=[r['record'] for e,r in selected],
        stream_forward_consistency=True,inference_device='cpu',shape=list(raw.shape),
        preprocessing='existing original causal H5; no additional processing',
        aggregation='six individually selected waveform records; descriptive case studies',
        output_order=['magnitude','distance_km','sin_azimuth','cos_azimuth'],summaries=summaries)
    (out/'results.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(out,flush=True)


if __name__=='__main__': main()
