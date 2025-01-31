from jupyterlab.semver import valid
from sklearn.cross_decomposition import CCA

import os
import numpy as np
import matplotlib
# if X is tunneled through ssh then use tkagg if headless use agg
if os.environ.get('DISPLAY', '') == '':
    matplotlib.use('Agg')
else:
    matplotlib.use('TkAgg')

from matplotlib import pyplot as plt
import sys
sys.path.append('../')
from config_lorenzres import interim_res_path, N, train_split, data_path_template, valid_split
from cdriver.preprocessing.splitters import train_valid_test_split
from cdriver.savers.saver import save_results
from cdriver.evaluate.evalz import comp_ccorr, get_maxes
from tqdm import tqdm
os.makedirs(interim_res_path, exist_ok=True)


plt.ion()
plt.figure(figsize=(10, 10))
plt.show()
mngr = plt.get_current_fig_manager()
# if tkagg backend is used then use wm_geometry else use set_position
if os.environ.get('DISPLAY', '') == '':
    pass
else:
    mngr.window.wm_geometry("+%d+%d" % (0, 0))

plt.xlim(-1, N)
plt.ylim(0, 1)

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

    plt.plot(n_iter, maxcs[-1], 'o', color='blue')
    plt.draw()
    plt.pause(0.05)

df = save_results(fname=interim_res_path / 'cca_res.csv',
                  r=maxcs,
                  N=N,
                  method='CCA',
                  dataset='lorenz')


plt.figure()
mngr = plt.get_current_fig_manager()
if os.environ.get('DISPLAY', '') == '':
    pass
else:
    mngr.window.wm_geometry("+%d+%d" % (1000, 0))
plt.hist(maxcs)
plt.xlim(0, 1)
plt.show()

plt.pause(2)
plt.close()

