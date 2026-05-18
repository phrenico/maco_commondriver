import argparse
from pathlib import Path

import os
import numpy as np
from sklearn.cross_decomposition import CCA
from cdriver.preprocessing.splitters import train_valid_test_split
from cdriver.savers.saver import save_results
from cdriver.evaluate.evalz import comp_ccorr, get_maxes
from scripts.experiments.config_loader import get_config, resolve_paths
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[3]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None)
    args = parser.parse_args()
    cfg = resolve_paths(get_config('lorenz', args.config), _REPO_ROOT)

    N = cfg['data']['N']
    data_path_template = str(_REPO_ROOT / cfg['data']['data_path_template'])
    train_split = cfg['preprocessing']['train_split']
    valid_split = cfg['preprocessing']['valid_split']
    n_components = cfg['methods']['cca']['n_components']
    max_iter = cfg['methods']['cca']['max_iter']
    interim_res_path = cfg['paths']['interim_res_path']
    os.makedirs(interim_res_path, exist_ok=True)

    maxcs = []
    for n_iter in tqdm(range(N)):
        data_path = data_path_template.format(n_iter)
        data = np.load(data_path)

        X = data['v'][:, 3:6]
        Y = data['v'][:, 6:]
        z = data['v'][:, 1]

        X_train, Y_train, z_train, X_valid, Y_valid, z_valid, X_test, Y_test, z_test = train_valid_test_split(X, Y, z,
                                                                                                              train_split,
                                                                                                              valid_split)

        cca = CCA(n_components=n_components, max_iter=max_iter)
        cca.fit(X_train, Y_train)

        zpred, zpred2 = cca.transform(X_test, Y_test)

        z_pred_m = (zpred[:, 0] + zpred2[:, 0]) / 2
        maxcs.append(get_maxes(*comp_ccorr(z_test, z_pred_m))[1])

    df = save_results(fname=interim_res_path / 'cca_res.csv',
                      r=maxcs,
                      N=N,
                      method='CCA',
                      dataset='lorenz')
