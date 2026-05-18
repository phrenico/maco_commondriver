import argparse
from pathlib import Path

import numpy as np
from sklearn.cross_decomposition import CCA
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

    # 2. Run CCA
    d_embed = cfg['preprocessing']['d_embed']
    train_split = cfg['preprocessing']['train_split']
    valid_split = cfg['preprocessing']['valid_split']
    n_components = cfg['methods']['cca']['n_components']

    maxcs = []
    maxcs2 = []
    maxcs3 = []
    for i in tqdm(range(N)):
        data = dataset[i]

        z = data[:-(d_embed - 1), 0]
        X = time_delay_embedding(data[:, 1], dimension=d_embed)
        Y = time_delay_embedding(data[:, 2], dimension=d_embed)

        X_train, Y_train, z_train, X_valid, Y_valid, z_valid, X_test, Y_test, z_test = train_valid_test_split(X, Y, z, train_split, valid_split)

        cca = CCA(n_components=n_components)
        cca.fit(X_train, Y_train)

        zpred, zpred2 = cca.transform(X_test, Y_test)

        tau, c = comp_ccorr(zpred[:, 0], z_test)
        tau2, c2 = comp_ccorr(zpred2[:, 0], z_test)
        tau3, c3 = comp_ccorr((zpred[:, 0] + zpred2[:, 0]) / 2, z_test)

        maxcs.append(get_maxes(tau, c)[1])
        maxcs2.append(get_maxes(tau2, c2)[1])
        maxcs3.append(get_maxes(tau3, c3)[1])

    # Save results
    df = save_results(fname=cfg['paths']['interim_res_path'] / 'cca_res.csv',
                      r=maxcs3,
                      N=N,
                      method='CCA',
                      dataset='logmap')
