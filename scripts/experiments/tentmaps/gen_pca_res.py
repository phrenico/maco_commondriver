'''Apply PCA to tentmap data — thin wrapper around method_runner.'''
import argparse

from scripts.experiments.method_runner import run_baseline_method
from sklearn.decomposition import PCA


def model_factory(method_cfg):
    return PCA(n_components=method_cfg['n_components'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='scripts/config_runall.py')
    args = parser.parse_args()
    run_baseline_method('tentmaps', 'PCA', model_factory, config_path=args.config)
