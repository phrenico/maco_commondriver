import numpy as np
from matplotlib import pyplot as plt

import sys
sys.path.append('../')

from data_generators import time_delay_embedding, comp_ccorr, get_maxes, train_test_split, save_results
from tqdm import tqdm

import matplotlib

import dca
DCA = dca.DynamicalComponentsAnalysis
matplotlib.use('TkAgg')

print(DCA.__name__)

plt.ion()
plt.figure(figsize=(10, 10))
plt.show()
mngr = plt.get_current_fig_manager()
mngr.window.wm_geometry("+%d+%d" % (0, 0))

plt.xlim(-1, 100)
plt.ylim(0, 1)
N = 100
train_split = 0.5
maxcs = []
for n_iter in tqdm(range(N)):
    data_path = '../../../data/lorenz/lorenz_{}.npz'.format(n_iter)
    data = np.load(data_path)

    X = data['v'][:, 3:6]
    Y = data['v'][:, 6:]
    z = data['v'][:, 1]

    train_split = 0.5
    X_train, Y_train, z_train, X_test, Y_test, z_test = train_test_split(X, Y, z, train_split)

    D_train = np.concatenate([X_train, Y_train], axis=1)
    D_test = np.concatenate([X_test, Y_test], axis=1)

    # 2. Run DCA
    n_components = 1
    dca_model = DCA(d=n_components, T=5, n_init=10)
    dca_model.fit(D_train)
    z_pred = dca_model.transform(D_test).squeeze()

    maxcs.append(get_maxes(*comp_ccorr(z_test, z_pred))[1])

    plt.plot(n_iter, maxcs[-1], 'o', color='blue')
    plt.draw()
    plt.pause(0.05)

df = save_results(fname='./dca_res.csv', r=maxcs, N=N, method='DCA', dataset='lorenz')

plt.ioff()
plt.figure()
mngr = plt.get_current_fig_manager()
mngr.window.wm_geometry("+%d+%d" % (1000, 0))
plt.hist(maxcs)