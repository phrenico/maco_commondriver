'''Apply DCA to tentmap data — thin wrapper.'''
import argparse
from scripts.experiments.method_runner import run_dca_method

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='scripts/config_runall.py')
    args = parser.parse_args()
    run_dca_method('tentmaps', config_path=args.config)
