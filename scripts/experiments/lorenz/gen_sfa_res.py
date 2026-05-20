'''Apply SFA to Lorenz data — thin wrapper.'''
import argparse
from scripts.experiments.method_runner import run_sfa_method

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None)
    args = parser.parse_args()
    run_sfa_method('lorenz', config_path=args.config)
