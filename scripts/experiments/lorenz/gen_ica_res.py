import argparse
from pathlib import Path

import os
import numpy as np
from sklearn.decomposition import FastICA
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
    n_components = cfg['methods']['ica']['n_components']
    interim_res_path = cfg['paths']['interim_res_path']
    os.makedirs(interim_res_path, exist_ok=True)

    maxcs = []
    for n_iter in tqdm(range(N)):
        data_path = data_path_template.format(n_iter)
        data = np.load(data_path)

        X = data['v'][:, 3:]
        z = data['v'][:, 1]
        T = X.shape[0]

        X_train, Y_train, z_train, X_valid, Y_valid, z_valid, X_test, Y_test, z_test = train_valid_test_split(X, X, z,
                                                                                                              train_split,
                                                                                                              valid_split)

        ica = FastICA(n_components=n_components).fit(X_train)
        zpred = ica.transform(X_test)

        m = max([get_maxes(*comp_ccorr(z_test, zpred[:, j]))[1] for j in range(n_components)])
        maxcs.append(m)

    df = save_results(fname=interim_res_path / 'ica_res.csv', r=maxcs, N=N, method='ICA', dataset='lorenz')
