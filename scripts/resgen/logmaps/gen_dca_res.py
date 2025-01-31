'''Run experiments with DCA

'''
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
# import dca
# # from dca import DynamicalComponentsAnalysis as
# DCA = dca.DynamicalComponentsAnalysis

import sys
sys.path.append('./')
sys.path.append('../../../')
from dca import DynamicalComponentsAnalysis as DCA

from cdriver.preprocessing.splitters import train_valid_test_split
from cdriver.preprocessing.tde import time_delay_embedding
from cdriver.savers.saver import save_results
from cdriver.evaluate.evalz import comp_ccorr, get_maxes
from cdriver.datagen.logmap import gen_logmapdata

from scripts.datagen_scripts.datagen_config import logmapgen_params
from config_logmapres import train_split, interim_res_path, valid_split


# 1. Generate data
N = logmapgen_params['N']  # number of realizations
dataset, params = gen_logmapdata(logmapgen_params)

d_embed = 3
maxcs = []
for i in tqdm(range(N)):
    data = dataset[i]

    z = data[:-(d_embed - 1), 0]
    X = time_delay_embedding(data[:, 1], dimension=d_embed)
    Y = time_delay_embedding(data[:, 2], dimension=d_embed)

    X_train, Y_train, z_train, X_valid, Y_valid, z_valid, X_test, Y_test, z_test = train_valid_test_split(X, Y, z, train_split, valid_split)
    D_train = np.concatenate([X_train, Y_train], axis=1)
    D_test = np.concatenate([X_test, Y_test], axis=1)

    # 2. Run DCA
    n_components = 1
    dca_model = DCA(d=n_components, T=5, n_init=10)
    dca_model.fit(D_train)
    zpred = dca_model.transform(D_test)  # .squeeze()

    m = max([get_maxes(*comp_ccorr(z_test, zpred[:, j]))[1] for j in range(n_components)])
    maxcs.append(m)

    # tau, c = comp_ccorr(zpred, z_test)
    # maxcs.append(get_maxes(tau, c)[1])

# Save results
df = save_results(fname=interim_res_path / 'dca_res.csv',
                  r=maxcs,
                  N=N,
                  method='DCA',
                  dataset='logmap')

# # 3. Plot results
# plt.figure()
# plt.hist(maxcs)
# plt.show()
