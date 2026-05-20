"""Run AniSOM on the logmap data and comparison.

NOTE: Intentionally NOT migrated to method_runner — AniSOM has a custom PyTorch training
loop (ani.fit() → ani.predict()) that doesn't fit the sklearn-style runner pattern.
"""
import os

os.environ['MPLBACKEND'] = 'Agg'
os.environ.setdefault('MPLCONFIGDIR', '/tmp')

import argparse
from pathlib import Path

import numpy as np
import torch

from cdriver.network.anisom import AniSOM
from cdriver.preprocessing.splitters import train_valid_test_split
from cdriver.preprocessing.tde import time_delay_embedding
from cdriver.savers.saver import save_results
from cdriver.evaluate.evalz import comp_ccorr, get_maxes
from cdriver.datagen.logmap import gen_logmapdata
from scripts.experiments.config_loader import get_config, resolve_paths
from tqdm import tqdm


def myfun(x, *args, **kwargs):
    return torch.linalg.eigh(x)

torch.symeig = myfun

_REPO_ROOT = Path(__file__).resolve().parents[3]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None)
    args = parser.parse_args()
    cfg = resolve_paths(get_config('logmaps', args.config), _REPO_ROOT)

    # 1. Generate data
    N = cfg['datagen']['N']
    dataset, params = gen_logmapdata(cfg['datagen'])

    d_embed = cfg['methods']['anisom']['d_embed']
    d_grid = cfg['methods']['anisom']['d_grid']
    sizes = cfg['methods']['anisom']['sizes']
    epochs = cfg['methods']['anisom']['epochs']
    train_split = cfg['preprocessing']['train_split']
    valid_split = cfg['preprocessing']['valid_split']

    maxcs = []
    for i in tqdm(range(N)):
        data = dataset[i]

        z = data[:-(d_embed - 1), 0]
        X = time_delay_embedding(data[:, 1], dimension=d_embed)
        Y = time_delay_embedding(data[:, 2], dimension=d_embed)

        X_train, Y_train, z_train, X_valid, Y_valid, z_valid, X_test, Y_test, z_test = train_valid_test_split(X, Y, z, train_split, valid_split)

        ani = AniSOM(space_dim=d_embed, grid_dim=d_grid, sizes=sizes)
        ani.fit(torch.Tensor(X_train), torch.Tensor(Y_train), epochs=epochs, disable_tqdm=True)
        pred = ani.predict(torch.Tensor(X_test))

        tau, c = comp_ccorr(pred[:, 1], z_test)
        maxcs.append(get_maxes(tau, c)[1])

    # Save results
    df = save_results(fname=cfg['paths']['interim_res_path'] / 'anisom_res.csv',
                      r=maxcs,
                      N=N,
                      method='ASOM',
                      dataset='logmap',
                      times=N * ['NaN'])