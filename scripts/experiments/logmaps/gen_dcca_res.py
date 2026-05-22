"""Run DCCA on logistic map data-set.

NOTE: Intentionally NOT migrated to method_runner — DCCA uses mvlearn with list-input
fit/transform, a torch.symeig monkey-patch, and per-realization model instantiation
that doesn't fit the generic sklearn-style runner.
"""
import os

os.environ['MPLBACKEND'] = 'Agg'
os.environ.setdefault('MPLCONFIGDIR', '/tmp')

import argparse
from pathlib import Path

import torch
torch.device('cpu')

import numpy as np
from tqdm import tqdm
from mvlearn.embed import DCCA
from cdriver.preprocessing.splitters import train_valid_test_split
from cdriver.preprocessing.tde import time_delay_embedding
from cdriver.savers.saver import save_results
from cdriver.evaluate.evalz import comp_ccorr, get_maxes
from cdriver.datagen.logmap import gen_logmapdata
from scripts.experiments.config_loader import get_config, resolve_paths


def myfun(x, *args, **kwargs):
    return torch.linalg.eigh(x)


torch.symeig = myfun  # redefine function to make it work with dcca

_REPO_ROOT = Path(__file__).resolve().parents[3]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='scripts/config_runall.py')
    args = parser.parse_args()
    cfg = resolve_paths(get_config('logmaps', args.config), _REPO_ROOT)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # 1. Generate data
    N = cfg['datagen']['N']
    dataset, params = gen_logmapdata(cfg['datagen'])

    # Define parameters and layers for deep model
    d_embed = cfg['preprocessing']['d_embed']
    train_split = cfg['preprocessing']['train_split']
    valid_split = cfg['preprocessing']['valid_split']
    features1, features2 = cfg['methods']['dcca']['features']
    layers1 = cfg['methods']['dcca']['layers']
    layers2 = layers1.copy()

    maxcs3 = []
    for i in tqdm(range(N)):
        data = dataset[i]

        z = data[:-(d_embed - 1), 0]
        X = time_delay_embedding(data[:, 1], dimension=d_embed)
        Y = time_delay_embedding(data[:, 2], dimension=d_embed)

        X_train, Y_train, z_train, X_valid, Y_valid, z_valid, X_test, Y_test, z_test = train_valid_test_split(X, Y, z, train_split, valid_split)

        dcca = DCCA(input_size1=features1, input_size2=features2, n_components=1,
                    layer_sizes1=layers1, layer_sizes2=layers2, epoch_num=500,
                    use_all_singular_values=True, device=device)
        dcca.fit([X_train, Y_train])
        Xs_transformed = dcca.transform([X_test, Y_test])

        zp1, zp2 = Xs_transformed
        tau, c3 = comp_ccorr((zp1[:, 0] + zp2[:, 0]) / 2, z_test)
        maxcs3.append(get_maxes(tau, c3)[1])

    # Save results
    df = save_results(fname=cfg['paths']['interim_res_path'] / 'dcca_res.csv',
                      r=maxcs3,
                      N=N,
                      method='DCCA',
                      dataset='logmap')
