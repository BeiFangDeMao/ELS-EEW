# -*- coding: utf-8 -*-
"""IDE: run directly. Full test runs only after minimal causal validation passes.

--validate-only : minimal 16-record streaming/full-window and geometry checks
--run-dir PATH : continue an already validated result directory (no overwrites)
--statistics-only PATH : recompute statistics from saved prediction arrays
"""
import os
for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(variable,'4')
import argparse
from datetime import datetime
from pathlib import Path
import hashlib

from config import HERE,WEIGHTS
from inference import validate_small,infer,dump
from analysis_metrics import analyze,validate_decisions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--validate-only',action='store_true')
    parser.add_argument('--run-dir',type=Path)
    parser.add_argument('--statistics-only',type=Path)
    args = parser.parse_args()
    if args.statistics_only:
        out = args.statistics_only.resolve()
        assert (out/'inference_manifest.json').exists()
        analyze(out)
        dump(HERE/'latest_result.json',dict(result_directory=str(out),stage='complete_record_level'))
        return
    out = args.run_dir.resolve() if args.run_dir else HERE/'results'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True,exist_ok=bool(args.run_dir))
    assert not (out/'predictions').exists(), 'Refusing to overwrite inference files'
    print('RESULT_DIRECTORY='+str(out),flush=True)
    initial_hash = hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()
    validate_decisions()
    validate_small(out)
    dump(HERE/'latest_result.json',dict(result_directory=str(out),stage='minimal_validation_passed'))
    if args.validate_only: return
    infer(out)
    assert hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()==initial_hash
    analyze(out)
    dump(HERE/'latest_result.json',dict(result_directory=str(out),stage='complete'))
    print('COMPLETE='+str(out),flush=True)


if __name__=='__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
