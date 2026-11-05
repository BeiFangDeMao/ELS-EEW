"""Reaudit saved original metadata without reading or inferring waveforms again."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from inference import audit_metadata,export_predictions,dump


def refresh(out):
    meta = pd.read_parquet(out/'predictions/records.parquet')
    meta,audit = audit_metadata(meta)
    meta.to_parquet(out/'predictions/records.parquet',index=False)
    meta.to_csv(out/'record_metadata.csv',index=False,encoding='utf-8-sig')
    dump(out/'metadata_audit.json',audit)
    manifest = json.loads((out/'inference_manifest.json').read_text(encoding='utf-8'))
    manifest['metadata_limits'] = audit['limitations']
    dump(out/'inference_manifest.json',manifest)
    folder = out/'predictions'
    export_predictions(out,meta,np.load(folder/'truths.npy',mmap_mode='r'),
        np.load(folder/'raw_predictions.npy',mmap_mode='r'),np.load(folder/'processed_predictions.npy',mmap_mode='r'),
        np.load(folder/'location_errors_km.npy',mmap_mode='r'))
    print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('result',type=Path)
    refresh(parser.parse_args().result)
