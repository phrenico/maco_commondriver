import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
sys.path.append('../')
from data_generators import LogmapExpRunner, comp_ccorr, get_maxes, train_test_split, save_results, time_delay_embedding
from shrec.models import RecurrenceManifold
from sklearn.preprocessing import scale
from scipy.signal import correlate, correlation_lags


import matplotlib

matplotlib.use('TkAgg')


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
for n_iter in range(N):
    data_path = '../../../data/lorenz/lorenz_{}.npz'.format(n_iter)
    data = np.load(data_path)

    X = data['v'][:, 3:]
    z = data['v'][:, 1]
    T = X.shape[0]

    X_train, Y_train, z_train, X_test, Y_test, z_test = train_test_split(X, X, z, train_split)

    d_embed = 1
    model = RecurrenceManifold(d_embed=d_embed)
    z_pred = model.fit_predict(X_test)

    maxcs.append(get_maxes(*comp_ccorr(z_test, z_pred))[1])
    plt.plot(n_iter, maxcs[-1], 'o', color='blue')
    plt.draw()
    plt.pause(0.05)

df = save_results(fname='./shrec_res.csv', r=maxcs, N=N, method='ShRec', dataset='lorenz')

plt.ioff()
plt.figure()
mngr = plt.get_current_fig_manager()
mngr.window.wm_geometry("+%d+%d" % (1000, 0))
plt.hist(maxcs)
plt.xlim(0, 1)
plt.show()
