"""Verify saved record-level predictions and irreversible classification metrics."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import confusion_matrix
from config import WEIGHTS
from inference import dump
from analysis_metrics import condition,strategy_specs,metrics,validate_decisions


def main(out):
    validate_decisions()
    records = pd.read_parquet(out/'predictions/records.parquet')
    raw = np.load(out/'predictions/raw_predictions.npy',mmap_mode='r')
    pred = np.load(out/'predictions/processed_predictions.npy',mmap_mode='r')
    truth = np.load(out/'predictions/truths.npy')
    errors = np.load(out/'predictions/location_errors_km.npy',mmap_mode='r')
    n,t = pred.shape[:2]
    assert n==51772 and t==93
    names = pq.ParquetFile(out/'predictions/predictions.parquet').schema_arrow.names
    assert not any(name in names for name in ('event_id','station_id','provided_event_id','event_group_id'))
    offset = 0
    columns = ['record_id','chunk','predicted_magnitude','predicted_distance_km','predicted_sin_azimuth',
               'predicted_cos_azimuth','raw_predicted_magnitude','location_error_km']
    for batch in pq.ParquetFile(out/'predictions/predictions.parquet').iter_batches(batch_size=131072,columns=columns):
        df = batch.to_pandas()
        indices = np.arange(offset,offset+len(df))
        ridx,cidx = indices//t,indices%t
        assert np.array_equal(df.record_id.to_numpy(),records.record_id.to_numpy()[ridx])
        assert np.array_equal(df.chunk.to_numpy(),cidx+1)
        for j,name in enumerate(('magnitude','distance_km','sin_azimuth','cos_azimuth')):
            assert np.array_equal(df['predicted_'+name].to_numpy(),pred[ridx,cidx,j])
        assert np.array_equal(df.raw_predicted_magnitude.to_numpy(),raw[ridx,cidx,0])
        assert np.array_equal(df.location_error_km.to_numpy(),errors[ridx,cidx])
        offset += len(df)
    assert offset==n*t
    warning = pd.read_csv(out/'warning_metrics.csv')
    alerts = pd.read_parquet(out/'alert_times.parquet')
    assert len(warning)==9*t and len(alerts)==9*n and set(warning.scope)=={'record_level'}
    for spec in strategy_specs():
        df = warning[warning.strategy==spec['name']].sort_values('chunk')
        actual = condition(spec,truth[:,0],truth[:,1])
        current = condition(spec,pred[:,:,0],pred[:,:,1])
        accumulated = np.maximum.accumulate(current,axis=1)
        assert np.all(np.diff(df.TP)>=0) and np.all(np.diff(df.FP)>=0)
        assert np.allclose(df.TP+df.FP+df.TN+df.FN,n)
        for j in (0,2,5,8,14,30,61,92):
            tn,fp,fn,tp = confusion_matrix(actual,accumulated[:,j],labels=[False,True]).ravel()
            row = df.iloc[j]
            assert (tn,fp,fn,tp)==(row.TN,row.FP,row.FN,row.TP)
            manual = metrics(np.array([tp],float),np.array([fp],float),np.array([tn],float),np.array([fn],float))
            for name in ('accuracy','precision','recall','f1','specificity','false_positive_rate','false_negative_rate','balanced_accuracy','mcc'):
                assert np.allclose(manual[name],row[name],equal_nan=True)
        af = alerts[alerts.strategy==spec['name']]
        expected = np.where(current.any(axis=1),current.argmax(axis=1)+1,np.nan)
        assert np.allclose(af.first_alert_chunk.to_numpy(),expected,equal_nan=True)
        assert af.record_id.tolist()==records.record_id.tolist()
    regression = pd.read_csv(out/'regression_metrics.csv')
    tolerance = pd.read_csv(out/'tolerance_rates.csv')
    assert len(regression)==744 and len(tolerance)==3627
    sizes = regression[(regression.task=='location_km')&(regression.chunk==1)].set_index('group').records
    assert sizes.drop('all').sum()==n
    assert (tolerance.successes<=tolerance.records).all()
    assert ((tolerance.ci95_low<=tolerance.rate+1e-12)&(tolerance.ci95_high>=tolerance.rate-1e-12)).all()
    mapping = pd.read_csv(out/'report_time_mapping.csv')
    assert (mapping.max_actual_time_after_p_s<=mapping.requested_time_s+1e-10).all()
    manifest = json.loads((out/'inference_manifest.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()==manifest['weight_sha256']
    quality = dict(analysis_unit='waveform_record',event_or_station_ids_computed=False,
        checked_prediction_rows=offset,warning_rows=len(warning),alert_rows=len(alerts),
        regression_rows=len(regression),tolerance_rows=len(tolerance),
        all_parquet_values_match_saved_predictions=True,confusion_matrices_independently_verified=True,
        earliest_alerts_preserved=True,irreversible_counts_monotonic=True,
        disjoint_distance_groups_cover_all_records=True,report_times_do_not_use_future_outputs=True,
        original_weight_sha256_unchanged=True)
    dump(out/'quality_checks.json',quality)
    print(json.dumps(quality,indent=2))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('result',type=Path)
    main(parser.parse_args().result)
