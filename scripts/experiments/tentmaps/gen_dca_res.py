'''Script to run DCA on tent map data-set'''
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm
from dca import DynamicalComponentsAnalysis as DCA

from cdriver.preprocessing.splitters import train_valid_test_split
from cdriver.preprocessing.tde import time_delay_embedding
from cdriver.savers.saver import save_results
from cdriver.evaluate.evalz import comp_ccorr, get_maxes
from cdriver.datagen.tent_map import gen_tentmapdata
from scripts.experiments.config_loader import get_config, resolve_paths

_REPO_ROOT = Path(__file__).resolve().parents[3]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None)
    args = parser.parse_args()
    cfg = resolve_paths(get_config('tentmaps', args.config), _REPO_ROOT)

    N = cfg['datagen']['N']
    dataset, params = gen_tentmapdata(cfg['datagen'])

    d_embed = cfg['methods']['dca'].get('d_embed', cfg['preprocessing']['d_embed'])
    train_split = cfg['preprocessing']['train_split']
    valid_split = cfg['preprocessing']['valid_split']
    n_components = cfg['methods']['dca']['n_components']
    T = cfg['methods']['dca']['T']
    n_init = cfg['methods']['dca']['n_init']

    maxcs = []
    for n_iter in tqdm(range(N)):
        data = dataset[n_iter]

        z = data[:-(d_embed - 1), 0]
        X = time_delay_embedding(data[:, 1], dimension=d_embed)
        Y = time_delay_embedding(data[:, 2], dimension=d_embed)

        X_train, Y_train, z_train, X_valid, Y_valid, z_valid, X_test, Y_test, z_test = train_valid_test_split(X, Y, z, train_split, valid_split)
        D_train = np.concatenate([X_train, Y_train], axis=1)
        D_test = np.concatenate([X_test, Y_test], axis=1)

        dca_model = DCA(d=n_components, T=T, n_init=n_init)
        dca_model.fit(D_train)
        z_pred = dca_model.transform(D_test)

        m = max([get_maxes(*comp_ccorr(z_test, z_pred[:, j]))[1] for j in range(n_components)])
        maxcs.append(m)

    df = save_results(fname=cfg['paths']['interim_res_path'] / 'dca_res.csv',
                      r=maxcs, N=N, method='DCA', dataset='tentmap')
