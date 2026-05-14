from jupyterlab.semver import valid
from mvlearn.embed import DCCA
import torch

import os
import numpy as np
from scripts.experiments.lorenz.config_lorenzres import interim_res_path, N, train_split, data_path_template, valid_split
from cdriver.preprocessing.splitters import train_valid_test_split
from cdriver.savers.saver import save_results
from cdriver.evaluate.evalz import comp_ccorr, get_maxes
from tqdm import tqdm
os.makedirs(interim_res_path, exist_ok=True)

def myfun(x, *args, **kwargs):
  return torch.linalg.eigh(x)

torch.symeig = myfun

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

maxcs = []
for n_iter in tqdm(range(N)):
    data_path = data_path_template.format(n_iter)
    data = np.load(data_path)

    X = data['v'][:, 3:6]
    Y = data['v'][:, 6:]
    z = data['v'][:, 1]


    d_embed = 3
    features1 = d_embed  # Feature sizes
    features2 = d_embed
    layers1 = [20, 20, 1]  # nodes in each hidden layer and the output size
    layers2 = layers1.copy()

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

df = save_results(fname=interim_res_path / './dcca_res.csv',
                  r=maxcs,
                  N=N,
                  method='DCCA',
                  dataset='lorenz')