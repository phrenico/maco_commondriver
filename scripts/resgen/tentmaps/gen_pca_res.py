import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import scale
import sys
sys.path.append('../')
sys.path.append('../../../')

from cdriver.preprocessing.splitters import train_valid_test_split
from cdriver.preprocessing.tde import time_delay_embedding
from cdriver.savers.saver import save_results
from cdriver.evaluate.evalz import comp_ccorr, get_maxes
from cdriver.datagen.tent_map import gen_tentmapdata

from scripts.datagen_scripts.datagen_config import tentmapgen_params
from config_tentmapres import train_split, interim_res_path, valid_split
import matplotlib

matplotlib.use('TkAgg')


plt.ion()
plt.figure(figsize=(10, 10))
plt.show()
mngr = plt.get_current_fig_manager()
mngr.window.wm_geometry("+%d+%d" % (0, 0))
plt.xlim(-1, tentmapgen_params['N'] + 1)
plt.ylim(0, 1)

N = tentmapgen_params['N']  # number of realizations
dataset, params = gen_tentmapdata(tentmapgen_params)


d_embed = 2
maxcs = []
for n_iter in range(N):
    data = dataset[n_iter]

    z = data[:-(d_embed - 1), 0]
    X = time_delay_embedding(data[:, 1], dimension=d_embed)
    Y = time_delay_embedding(data[:, 2], dimension=d_embed)

    X_train, Y_train, z_train, X_valid, Y_valid, z_valid, X_test, Y_test, z_test = train_valid_test_split(X, Y, z,
                                                                                                          train_split,
                                                                                                          valid_split)
    D_train = np.concatenate([X_train, Y_train], axis=1)
    D_test = np.concatenate([X_test, Y_test], axis=1)

    n_components = 1
    pca =  PCA(n_components=n_components).fit(D_train)
    zpred = pca.transform(D_test)

    m = max([get_maxes(*comp_ccorr(z_test, zpred[:, j]))[1] for j in range(n_components)])
    maxcs.append(m)
    plt.plot(n_iter, maxcs[-1], 'o', color='blue')
    plt.draw()
    plt.pause(0.05)

df = save_results(fname=interim_res_path / './pca_res.csv',
                  r=maxcs,
                  N=N,
                  method='PCA',
                  dataset='tentmap')

# plt.ioff()
plt.figure()
mngr = plt.get_current_fig_manager()
mngr.window.wm_geometry("+%d+%d" % (1000, 0))
plt.hist(maxcs)
plt.xlim(0, 1)
plt.show()
plt.close()
