'''Apply CCA to the Lorenz system — thin wrapper around method_runner.'''
import argparse

from scripts.experiments.method_runner import run_baseline_method
from sklearn.cross_decomposition import CCA


def model_factory(method_cfg):
    return CCA(n_components=method_cfg['n_components'], max_iter=method_cfg.get('max_iter', 500))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None)
    args = parser.parse_args()
    run_baseline_method('lorenz', 'CCA', model_factory, config_path=args.config)
