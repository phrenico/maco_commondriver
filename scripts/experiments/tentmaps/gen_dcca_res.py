import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import scale

from mvlearn.embed import DCCA
from cdriver.preprocessing.splitters import train_valid_test_split
from cdriver.preprocessing.tde import time_delay_embedding
from cdriver.savers.saver import save_results
from cdriver.evaluate.evalz import comp_ccorr, get_maxes
from cdriver.datagen.tent_map import gen_tentmapdata

from scripts.datagen_scripts.datagen_config import tentmapgen_params
from scripts.experiments.tentmaps.config_tentmapres import train_split, interim_res_path, valid_split
import torch
from tqdm import tqdm

import matplotlib
matplotlib.use('TkAgg')


def myfun(x, *args, **kwargs):
    return torch.linalg.eigh(x)


torch.symeig = myfun
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

plt.ion()
plt.figure(figsize=(10, 10))
mngr = plt.get_current_fig_manager()
mngr.window.wm_geometry("+%d+%d" % (0, 0))
plt.show()
plt.xlim(-1, tentmapgen_params['N'] + 1)
plt.ylim(0, 1)

N = tentmapgen_params['N']  # number of realizations
dataset, params = gen_tentmapdata(tentmapgen_params)

d_embed = 2
maxcs = []
for n_iter in tqdm(range(N)):
    data = dataset[n_iter]

    z = data[:-(d_embed - 1), 0]
    X = time_delay_embedding(data[:, 1], dimension=d_embed)
    Y = time_delay_embedding(data[:, 2], dimension=d_embed)

    features1 = d_embed  # Feature sizes
    features2 = d_embed
    layers1 = [20, 20, 1]  # nodes in each hidden layer and the output size
    layers2 = layers1.copy()

    X_train, Y_train, z_train, X_valid, Y_valid, z_valid, X_test, Y_test, z_test = train_valid_test_split(X, Y, z,
                                                                                                          train_split,
                                                                                                          valid_split)

    dcca = DCCA(input_size1=features1,
                input_size2=features2,
                n_components=1,
                layer_sizes1=layers1,
                layer_sizes2=layers2,
                epoch_num=100,
                use_all_singular_values=True, device=device)
    dcca.fit([X_train, Y_train])
    Xs_transformed = dcca.transform([X_test, Y_test])

    zp1, zp2 = Xs_transformed

    z_pred = (zp1[:, 0] + zp2[:, 0]) / 2
    maxcs.append(get_maxes(*comp_ccorr(z_test, z_pred))[1])

    plt.plot(n_iter, maxcs[-1], 'o', color='blue')
    plt.draw()
    plt.pause(0.05)

df = save_results(fname=interim_res_path / './dcca_res.csv',
                  r=maxcs,
                  N=N,
                  method='DCCA',
                  dataset='tentmap')

# plt.ioff()
plt.figure()
mngr = plt.get_current_fig_manager()
mngr.window.wm_geometry("+%d+%d" % (1000, 0))
plt.hist(maxcs)
plt.xlim(0, 1)
plt.show()
plt.close()
