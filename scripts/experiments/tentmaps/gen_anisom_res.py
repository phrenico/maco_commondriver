"""Run AniSOM on the tent map data and comparison.

NOTE: Intentionally NOT migrated to method_runner — see logmaps/gen_anisom_res.py.
"""
import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from cdriver.network.anisom import AniSOM
from cdriver.preprocessing.splitters import train_valid_test_split
from cdriver.preprocessing.tde import time_delay_embedding
from cdriver.savers.saver import save_results
from cdriver.evaluate.evalz import comp_ccorr, get_maxes
from cdriver.datagen.tent_map import gen_tentmapdata
from scripts.experiments.config_loader import get_config, resolve_paths


def myfun(x, *args, **kwargs):
    return torch.linalg.eigh(x)

torch.symeig = myfun

_REPO_ROOT = Path(__file__).resolve().parents[3]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None)
    args = parser.parse_args()
    cfg = resolve_paths(get_config('tentmaps', args.config), _REPO_ROOT)

    N = cfg['datagen']['N']
    dataset, params = gen_tentmapdata(cfg['datagen'])

    d_embed = cfg['methods']['anisom']['d_embed']
    d_grid = cfg['methods']['anisom']['d_grid']
    sizes = cfg['methods']['anisom']['sizes']
    epochs = cfg['methods']['anisom']['epochs']
    train_split = cfg['preprocessing']['train_split']
    valid_split = cfg['preprocessing']['valid_split']

    maxcs = []
    for n_iter in tqdm(range(N)):
        data = dataset[n_iter]

        z = data[:-(d_embed - 1), 0]
        X = time_delay_embedding(data[:, 1], dimension=d_embed)
        Y = time_delay_embedding(data[:, 2], dimension=d_embed)

        X_train, Y_train, z_train, X_valid, Y_valid, z_valid, X_test, Y_test, z_test = train_valid_test_split(X, Y, z, train_split, valid_split)

        ani = AniSOM(space_dim=d_embed, grid_dim=d_grid, sizes=sizes)
        ani.fit(torch.Tensor(X_train), torch.Tensor(Y_train), epochs=epochs, disable_tqdm=True)
        pred = ani.predict(torch.Tensor(X_test))

        tau, c = comp_ccorr(pred[:, 1], z_test)
        maxcs.append(get_maxes(tau, c)[1])

    df = save_results(fname=cfg['paths']['interim_res_path'] / 'anisom_res.csv',
                      r=maxcs, N=N, method='ASOM', dataset='tentmap',
                      times=N * ['NaN'])
