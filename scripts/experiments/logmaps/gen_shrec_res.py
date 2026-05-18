'''Run ShRec experiments on logistic map data-set'''
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm
from shrec.models import RecurrenceManifold
from cdriver.preprocessing.splitters import train_valid_test_split
from cdriver.preprocessing.tde import time_delay_embedding
from cdriver.savers.saver import save_results
from cdriver.evaluate.evalz import comp_ccorr, get_maxes
from cdriver.datagen.logmap import gen_logmapdata
from scripts.experiments.config_loader import get_config, resolve_paths

_REPO_ROOT = Path(__file__).resolve().parents[3]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None)
    args = parser.parse_args()
    cfg = resolve_paths(get_config('logmaps', args.config), _REPO_ROOT)

    N = cfg['datagen']['N']
    dataset, params = gen_logmapdata(cfg['datagen'])

    train_split = cfg['preprocessing']['train_split']
    valid_split = cfg['preprocessing']['valid_split']
    d_embed = cfg['methods']['shrec']['d_embed']

    maxcs = []
    for i in tqdm(range(N)):
        data = dataset[i]
        X = data[:, 1:]
        y = data[:, 0]

        X_train, _, z_train, X_valid, _, z_valid, X_test, __, z_test = train_valid_test_split(X, X, y, train_split, valid_split)

        model = RecurrenceManifold(d_embed=d_embed)
        y_recon = model.fit_predict(X_train)

        tau, c = comp_ccorr(z_train, y_recon)
        maxtau, maxc = get_maxes(tau, c)
        maxcs.append(maxc)

    # Save results
    df = save_results(fname=cfg['paths']['interim_res_path'] / 'shrec_res.csv',
                      r=maxcs,
                      N=N,
                      method='ShRec',
                      dataset='logmap')