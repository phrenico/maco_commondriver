'''Apply ShRec to Lorenz data — thin wrapper.'''
import argparse
from scripts.experiments.method_runner import run_shrec_method

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='scripts/config_runall.py')
    args = parser.parse_args()
    run_shrec_method('lorenz', config_path=args.config)
