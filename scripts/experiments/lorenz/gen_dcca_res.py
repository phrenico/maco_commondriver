# NOTE: Intentionally NOT migrated to method_runner — DCCA uses mvlearn with list-input
# fit/transform and a torch.symeig monkey-patch. See logmaps/gen_dcca_res.py.
import argparse
from pathlib import Path

import os
import torch
import numpy as np
from mvlearn.embed import DCCA
from cdriver.preprocessing.splitters import train_valid_test_split
from cdriver.savers.saver import save_results
from cdriver.evaluate.evalz import comp_ccorr, get_maxes
from scripts.experiments.config_loader import get_config, resolve_paths
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[3]


def myfun(x, *args, **kwargs):
    return torch.linalg.eigh(x)

torch.symeig = myfun


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='scripts/config_runall.py')
    args = parser.parse_args()
    cfg = resolve_paths(get_config('lorenz', args.config), _REPO_ROOT)

    N = cfg['data']['N']
    data_path_template = str(_REPO_ROOT / cfg['data']['data_path_template'])
    train_split = cfg['preprocessing']['train_split']
    valid_split = cfg['preprocessing']['valid_split']
    d_embed = cfg['methods']['dcca']['d_embed']
    layers = cfg['methods']['dcca']['layers']
    interim_res_path = cfg['paths']['interim_res_path']
    os.makedirs(interim_res_path, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    maxcs = []
    for n_iter in tqdm(range(N)):
        data_path = data_path_template.format(n_iter)
        data = np.load(data_path)

        X = data['v'][:, 3:6]
        Y = data['v'][:, 6:]
        z = data['v'][:, 1]

        features1 = d_embed
        features2 = d_embed
        layers1 = list(layers)
        layers2 = list(layers)

        X_train, Y_train, z_train, X_valid, Y_valid, z_valid, X_test, Y_test, z_test = train_valid_test_split(X, Y, z,
                                                                                                              train_split,
                                                                                                              valid_split)

        dcca = DCCA(input_size1=features1, input_size2=features2, n_components=1,
                    layer_sizes1=layers1, layer_sizes2=layers2, epoch_num=100,
                    use_all_singular_values=True, device=device)
        dcca.fit([X_train, Y_train])
        Xs_transformed = dcca.transform([X_test, Y_test])

        zp1, zp2 = Xs_transformed

        z_pred = (zp1[:, 0] + zp2[:, 0]) / 2
        maxcs.append(get_maxes(*comp_ccorr(z_test, z_pred))[1])

    df = save_results(fname=interim_res_path / 'dcca_res.csv',
                      r=maxcs,
                      N=N,
                      method='DCCA',
                      dataset='lorenz')