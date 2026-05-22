'''Random baseline for tentmap data — thin wrapper.'''
import argparse
from scripts.experiments.method_runner import run_random_baseline

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='scripts/config_runall.py')
    args = parser.parse_args()
    run_random_baseline('tentmaps', config_path=args.config)
