'''Script to run ICA on logistic map data-set
1. Generate data
2. Run ICA
3. Save results
'''
import argparse
from pathlib import Path

import numpy as np
from sklearn.decomposition import FastICA
from cdriver.preprocessing.splitters import train_valid_test_split
from cdriver.preprocessing.tde import time_delay_embedding
from cdriver.savers.saver import save_results
from cdriver.evaluate.evalz import comp_ccorr, get_maxes
from cdriver.datagen.logmap import gen_logmapdata
from scripts.experiments.config_loader import get_config, resolve_paths
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[3]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None)
    args = parser.parse_args()
    cfg = resolve_paths(get_config('logmaps', args.config), _REPO_ROOT)

    # 1. Generate data
    N = cfg['datagen']['N']
    dataset, params = gen_logmapdata(cfg['datagen'])

    # 2. Run ICA
    d_embed = cfg['preprocessing']['d_embed']
    train_split = cfg['preprocessing']['train_split']
    valid_split = cfg['preprocessing']['valid_split']
    n_components = cfg['methods']['ica']['n_components']

    maxcs = []
    for i in tqdm(range(N)):
        X = time_delay_embedding(dataset[i][:, 1], delay=1, dimension=d_embed)
        Y = time_delay_embedding(dataset[i][:, 2], delay=1, dimension=d_embed)
        z = dataset[i][d_embed - 1:, 0]
        X_train, Y_train, z_train, X_valid, Y_valid, z_valid, X_test, Y_test, z_test = train_valid_test_split(X, Y, z, train_split, valid_split)

        D = np.concatenate([X_train, Y_train], axis=1)

        model = FastICA(n_components=n_components).fit(D)
        zpred = model.transform(np.concatenate([X_test, Y_test], axis=1))

        m = max([get_maxes(*comp_ccorr(z_test, zpred[:, j]))[1] for j in range(n_components)])
        maxcs.append(m)

    # Save results
    df = save_results(fname=cfg['paths']['interim_res_path'] / 'ica_res.csv',
                      r=maxcs,
                      N=N,
                      method='ICA',
                      dataset='logmap')