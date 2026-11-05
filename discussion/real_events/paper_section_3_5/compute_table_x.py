"""Six-row Table X from the exact Fig13 bundle; no model, inference or plotting.

Localization error: WGS84 inverse between the saved predicted epicenter and
catalog epicenter, using the same GeographicLib convention as Fig13.
All times are seconds after the corresponding annotated P pick, not origin
time, automatic trigger time, network delivery time or S arrival.
M5 crossing is independent of the fixed 2.88 s decision-center initial report.
"""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
try:
    from geographiclib.geodesic import Geodesic
except ImportError:
    sys.path.insert(0, str(HERE.parent / 'plot_dependencies'))
    from geographiclib.geodesic import Geodesic


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    figure_manifest_path = ROOT / 'plotting/Fig13_Six_Event_Cases/manifest.json'
    figure = json.loads(figure_manifest_path.read_text(encoding='utf-8'))
    source = Path(figure['source_bundle'])
    protected = {str(figure_manifest_path): sha(figure_manifest_path)}
    for output in figure['outputs']:
        for extension in ('pdf', 'png'):
            path = Path(output[extension])
            protected[str(path)] = sha(path)
    for relative, expected in figure['source_sha256'].items():
        path = source / relative
        assert sha(path) == expected, f'Fig13 data no longer matches its manifest: {path}'
        protected[str(path)] = expected
    events = json.loads((source / 'events.json').read_text(encoding='utf-8'))
    expected_order = [('20240101','ISK003'),('20240808','MYZ013'),('20240417','EHM012'),
                      ('20230505','ISK002'),('20240315','FKS011'),('20230511','CHB027')]
    headers = ['event_id','event_name','station','true_magnitude',
               'magnitude_absolute_error_2_88s','magnitude_absolute_error_29_76s',
               'location_error_2_88s_km','location_error_29_76s_km','first_M5_time_after_P_s']
    rows, audit = [], []
    for event, (date, station_code) in zip(events, expected_order):
        station = event['stations'][0]
        assert event['date']==date and station['station_code']==station_code and station['rank']==1
        path = source / station['inference_json']
        data = json.loads(path.read_text(encoding='utf-8'))
        truth, meta, output = data['truth'], data['metadata'], data['outputs']
        assert len(output)==93 and [p['output_index'] for p in output]==list(range(1,94))
        assert meta['station_code']==station_code and meta['record_id']==station['record_id']
        if date=='20240101':
            assert meta['p_pick_seconds']==13.52, 'Confirmed current-Fig13 original P version required'
        assert data['model']['checkpoint_sha256']=='c20841d6c4d76776dae64b83847cb5e788d287f59f52adab592f183639557f80'
        nominal = np.array([p['nominal_time_after_p_s'] for p in output])
        time = np.array([p['exact_time_after_p_s'] for p in output])
        mag = np.array([p['magnitude'] for p in output])
        lon = np.array([p['predicted_longitude_deg'] for p in output])
        lat = np.array([p['predicted_latitude_deg'] for p in output])
        np.testing.assert_allclose(nominal, np.arange(1,94)*.32,rtol=0,atol=1e-10)
        np.testing.assert_allclose(time-nominal,meta['crop_alignment_offset_s'],rtol=0,atol=1e-10)
        # These are exactly the four arrays hashed by the Fig13 plotting script.
        signature = hashlib.sha256(np.column_stack([time,lon,lat,mag]).tobytes()).hexdigest()
        checks = [c for c in figure['checks'] if c['event_date']==date]
        assert len(checks)==4 and all(c['plotted_values_sha256']==signature for c in checks)
        assert np.isfinite(np.column_stack([time,mag,lon,lat])).all()
        errors, locations = [], []
        for index, requested in [(8,2.88),(92,29.76)]:
            assert abs(nominal[index]-requested)<1e-10 and abs(time[index]-requested)<1e-10
            errors.append(float(abs(mag[index]-truth['magnitude'])))
            locations.append(float(Geodesic.WGS84.Inverse(truth['epicenter_latitude_deg'],truth['epicenter_longitude_deg'],
                                                         float(lat[index]),float(lon[index]))['s12']/1000))
        crossing = np.flatnonzero(mag>=5.)
        first = int(crossing[0]) if len(crossing) else None
        first_time = float(time[first]) if first is not None else None
        if first is not None:
            assert mag[first]>=5 and not np.any(mag[:first]>=5)
        rows.append([date,event['english_name'],station_code,float(truth['magnitude']),*errors,*locations,first_time])
        audit.append(dict(event_id=date,station=station_code,record_id=meta['record_id'],
                          inference_json=str(path),inference_sha256=sha(path),
                          p_pick_seconds=meta['p_pick_seconds'],p_version=meta['p_version'],
                          output_count=93,fig13_plotted_arrays_sha256=signature,
                          first_M5_status='reached' if first is not None else 'not_reached',
                          first_M5_output_chunk=first+1 if first is not None else None,
                          first_M5_predicted_magnitude=float(mag[first]) if first is not None else None,
                          previous_predicted_magnitude=float(mag[first-1]) if first is not None and first>0 else None,
                          automatic_P_trigger=None,verified_S_pick=None,
                          missing_pick_reason='Not provided as automatic P trigger or verified S pick in the Fig13 bundle'))
    assert len(events)==len(rows)==6
    for path, expected in protected.items():
        assert sha(path)==expected, f'Unexpected change to source or figure: {path}'
    out = HERE / 'results' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True)
    payload = dict(headers=headers,rows=rows,audit=audit,source_bundle=str(source),
                   time_reference='seconds after each record annotated P pick; original Fig13 P version, Noto=13.52 s from record start',
                   magnitude_error_definition='absolute error of a single record at one output, not a multi-record MAE',
                   location_error_definition='WGS84 inverse geodesic distance in km from saved predicted epicenter to catalog epicenter',
                   M5_definition='first saved prediction >=5.0 in original 93-output order; no smoothing or persistence condition; null if never reached',
                   threshold_crossing_is_not_initial_report_time=True,initial_report_policy_s=2.88,
                   no_training=True,no_inference=True,no_new_figures=True,
                   protected_files_sha256=protected)
    (out/'table_x_data.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    (HERE/'latest_run.json').write_text(json.dumps(dict(result_directory=str(out)),indent=2),encoding='utf-8')
    print(json.dumps(dict(result_directory=str(out),headers=headers,rows=rows),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
