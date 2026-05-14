from sklearn.cross_decomposition import CCA

import os
import numpy as np
from scripts.experiments.lorenz.config_lorenzres import interim_res_path, N, train_split, data_path_template, valid_split
from cdriver.preprocessing.splitters import train_valid_test_split
from cdriver.savers.saver import save_results
from cdriver.evaluate.evalz import comp_ccorr, get_maxes
from tqdm import tqdm
os.makedirs(interim_res_path, exist_ok=True)

train_split = train_split
maxcs = []
for n_iter in tqdm(range(N)):
    data_path = data_path_template.format(n_iter)
    data = np.load(data_path)

    X = data['v'][:, 3:6]
    Y = data['v'][:, 6:]
    z = data['v'][:, 1]

    X_train, Y_train, z_train, X_valid, Y_valid, z_valid, X_test, Y_test, z_test = train_valid_test_split(X, Y, z,
                                                                                                          train_split,
                                                                                                          valid_split)

    cca = CCA(n_components=1, max_iter=500)
    cca.fit(X_train, Y_train)

    zpred, zpred2 = cca.transform(X_test, Y_test)

    z_pred_m = (zpred[:, 0] + zpred2[:, 0]) / 2
    maxcs.append(get_maxes(*comp_ccorr(z_test, z_pred_m))[1])

df = save_results(fname=interim_res_path / 'cca_res.csv',
                  r=maxcs,
                  N=N,
                  method='CCA',
                  dataset='lorenz')

