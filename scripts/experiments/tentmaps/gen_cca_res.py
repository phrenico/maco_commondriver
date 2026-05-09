import numpy as np

from sklearn.cross_decomposition import CCA

from sympy.physics.control.control_plots import matplotlib


from cdriver.preprocessing.splitters import train_valid_test_split
from cdriver.preprocessing.tde import time_delay_embedding
from cdriver.savers.saver import save_results
from cdriver.evaluate.evalz import comp_ccorr, get_maxes
from cdriver.datagen.tent_map import gen_tentmapdata

from scripts.datagen_scripts.datagen_config import tentmapgen_params
from scripts.experiments.tentmaps.config_tentmapres import train_split, interim_res_path, valid_split
import matplotlib.pyplot as plt

from tqdm import tqdm
import pandas as pd

import matplotlib
matplotlib.use('TkAgg')

# 1. Generate data
N = tentmapgen_params['N']  # number of realizations
dataset, params = gen_tentmapdata(tentmapgen_params)

plt.ion()
plt.figure(figsize=(10, 10))
plt.show()
mngr = plt.get_current_fig_manager()
mngr.window.wm_geometry("+%d+%d" % (0, 0))
plt.xlim(-1, N + 1)
plt.ylim(0, 1)

d_embed = 3

maxcs = []
maxcs2 = []
maxcs3 = []
for n_iter in tqdm(range(N)):
    data = dataset[n_iter]

    z = data[:-(d_embed - 1), 0]
    X = time_delay_embedding(data[:, 1], dimension=d_embed)
    Y = time_delay_embedding(data[:, 2], dimension=d_embed)

    X_train, Y_train, z_train, X_valid, Y_valid, z_valid, X_test, Y_test, z_test = train_valid_test_split(X, Y, z,
                                                                                                          train_split,
                                                                                                          valid_split)

    cca = CCA(n_components=1)
    cca.fit(X_train, Y_train)

    zpred, zpred2 = cca.transform(X_test, Y_test)

    tau, c = comp_ccorr(zpred[:, 0], z_test)
    tau2, c2 = comp_ccorr(zpred2[:, 0], z_test)
    tau3, c3 = comp_ccorr((zpred[:, 0] + zpred2[:, 0]) / 2, z_test)

    maxcs.append(get_maxes(tau, c)[1])
    maxcs2.append(get_maxes(tau2, c2)[1])
    maxcs3.append(get_maxes(tau3, c3)[1])

    plt.plot(n_iter, maxcs3[-1], 'o', color='blue')
    plt.draw()
    plt.pause(0.05)

# Save results
df = save_results(fname=interim_res_path / './cca_res.csv',
                  r=maxcs3,
                  N=N,
                  method='CCA',
                  dataset='tentmap')

# plt.ioff()
plt.figure()
plt.hist(maxcs)
plt.hist(maxcs2)
plt.hist(maxcs3)
plt.show()
plt.close()
