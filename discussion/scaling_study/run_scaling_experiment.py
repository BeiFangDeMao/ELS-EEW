"""Windows-safe entry: heavy imports happen only in the guarded main process."""
import argparse
import multiprocessing
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='ELS-EEW scaling: check or complete sequential experiment')
    parser.add_argument('--check', action='store_true', help='preflight + tiny synthetic checks; NEVER full training')
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('experiment_config.yaml'))
    parser.add_argument('--new-run', action='store_true', help='Start a new timestamped run instead of resuming the active run')
    args = parser.parse_args()
    from study import execute
    execute(args.config, check_only=args.check, new_run=args.new_run)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
