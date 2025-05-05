import numpy as np
import sys
sys.path.append('./')
sys.path.append('../../../')
from sklearn.cross_decomposition import CCA
from cdriver.preprocessing.splitters import train_valid_test_split
from cdriver.preprocessing.tde import time_delay_embedding
from cdriver.savers.saver import save_results
from cdriver.evaluate.evalz import comp_ccorr, get_maxes
from cdriver.datagen.logmap import gen_logmapdata

from scripts.datagen_scripts.datagen_config import logmapgen_params
from config_logmapres import train_split, interim_res_path, valid_split
import matplotlib.pyplot as plt

from tqdm import tqdm

# 1. Generate data
N = logmapgen_params['N']  # number of realizations
dataset, params = gen_logmapdata(logmapgen_params)

# 2. Run CCA
d_embed = 3

maxcs = []
maxcs2 = []
maxcs3 = []
for i in tqdm(range(N)):
    data = dataset[i]

    z = data[:-(d_embed - 1), 0]
    X = time_delay_embedding(data[:, 1], dimension=d_embed)
    Y = time_delay_embedding(data[:, 2], dimension=d_embed)

    X_train, Y_train, z_train, X_valid, Y_valid, z_valid, X_test, Y_test, z_test = train_valid_test_split(X, Y, z, train_split, valid_split)

    cca = CCA(n_components=1)
    cca.fit(X_train, Y_train)

    zpred, zpred2 = cca.transform(X_test, Y_test)

    tau, c = comp_ccorr(zpred[:, 0], z_test)
    tau2, c2 = comp_ccorr(zpred2[:, 0], z_test)
    tau3, c3 = comp_ccorr((zpred[:, 0] + zpred2[:, 0]) / 2, z_test)

    maxcs.append(get_maxes(tau, c)[1])
    maxcs2.append(get_maxes(tau2, c2)[1])
    maxcs3.append(get_maxes(tau3, c3)[1])

# Save results
df = save_results(fname=interim_res_path / 'cca_res.csv',
                  r=maxcs3,
                  N=N,
                  method='CCA',
                  dataset='logmap')

# plt.hist(maxcs3)
# plt.show()
