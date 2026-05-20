'''Random baseline for Lorenz data — thin wrapper.'''
import argparse
from scripts.experiments.method_runner import run_random_baseline

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None)
    args = parser.parse_args()
    run_random_baseline('lorenz', config_path=args.config)
