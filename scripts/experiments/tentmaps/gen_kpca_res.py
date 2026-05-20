'''Apply KPCA to tentmap data — thin wrapper around method_runner.'''
import argparse

from scripts.experiments.method_runner import run_baseline_method
from sklearn.decomposition import KernelPCA


def model_factory(method_cfg):
    return KernelPCA(n_components=method_cfg['n_components'], kernel=method_cfg['kernel'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None)
    args = parser.parse_args()
    run_baseline_method('tentmaps', 'KPCA', model_factory, config_path=args.config)
