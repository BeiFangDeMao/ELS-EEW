"""Unified IDE/Windows entry. Running without --check starts the real experiment."""
import argparse
import multiprocessing
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true',help='Short preflight and synthetic checks; no full training')
    parser.add_argument('--new-run',action='store_true',help='New timestamped results instead of active-run resume')
    parser.add_argument('--cache-only',action='store_true',help='Only existing 8K CPU cache consistency/latency experiment')
    parser.add_argument('--config',type=Path,default=Path(__file__).with_name('ablation_config.yaml'))
    args=parser.parse_args()
    from ablation_runner import execute
    execute(args.config,check_only=args.check,new_run=args.new_run,cache_only=args.cache_only)


if __name__=='__main__':
    multiprocessing.freeze_support()
    main()
