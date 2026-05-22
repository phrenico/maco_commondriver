'''Script to run DCCA on tent map data-set.

NOTE: Intentionally NOT migrated to method_runner — see logmaps/gen_dcca_res.py.
'''
import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from mvlearn.embed import DCCA

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
    parser.add_argument('--config', default='scripts/config_runall.py')
    args = parser.parse_args()
    cfg = resolve_paths(get_config('tentmaps', args.config), _REPO_ROOT)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    N = cfg['datagen']['N']
    dataset, params = gen_tentmapdata(cfg['datagen'])

    d_embed = cfg['preprocessing']['d_embed']
    train_split = cfg['preprocessing']['train_split']
    valid_split = cfg['preprocessing']['valid_split']
    features1, features2 = cfg['methods']['dcca']['features']
    layers1 = cfg['methods']['dcca']['layers']
    layers2 = layers1.copy()

    maxcs = []
    for n_iter in tqdm(range(N)):
        data = dataset[n_iter]

        z = data[:-(d_embed - 1), 0]
        X = time_delay_embedding(data[:, 1], dimension=d_embed)
        Y = time_delay_embedding(data[:, 2], dimension=d_embed)

        X_train, Y_train, z_train, X_valid, Y_valid, z_valid, X_test, Y_test, z_test = train_valid_test_split(X, Y, z, train_split, valid_split)

        dcca = DCCA(input_size1=features1, input_size2=features2, n_components=1,
                    layer_sizes1=layers1, layer_sizes2=layers2, epoch_num=100,
                    use_all_singular_values=True, device=device)
        dcca.fit([X_train, Y_train])
        Xs_transformed = dcca.transform([X_test, Y_test])

        zp1, zp2 = Xs_transformed
        z_pred = (zp1[:, 0] + zp2[:, 0]) / 2
        maxcs.append(get_maxes(*comp_ccorr(z_test, z_pred))[1])

    df = save_results(fname=cfg['paths']['interim_res_path'] / 'dcca_res.csv',
                      r=maxcs, N=N, method='DCCA', dataset='tentmap')
