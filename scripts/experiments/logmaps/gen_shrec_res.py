'''Apply ShRec to logmap data — thin wrapper.'''
import argparse
from scripts.experiments.method_runner import run_shrec_method

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None)
    args = parser.parse_args()
    run_shrec_method('logmaps', config_path=args.config)
