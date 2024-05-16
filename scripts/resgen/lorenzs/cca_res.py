import numpy as np
from matplotlib import pyplot as plt
# from sklearn.decomposition import PCA, FastICA
from sklearn.cross_decomposition import CCA

from sklearn.preprocessing import scale
import sys
sys.path.append('../')

from data_generators import time_delay_embedding, comp_ccorr, get_maxes, train_test_split, save_results, train_valid_test_split
from scipy.signal import correlate, correlation_lags
from tqdm import tqdm

# import matplotlib
#
# matplotlib.use('sgg')

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

    X_train, Y_train, z_train, X_test, Y_test, z_test = train_test_split(X, Y, z, train_split)

    cca = CCA(n_components=1, max_iter=500)
    cca.fit(X_train, Y_train)

    zpred, zpred2 = cca.transform(X_test, Y_test)

    z_pred = (zpred[:, 0] + zpred2[:, 0]) / 2
    maxcs.append(get_maxes(*comp_ccorr(z_test, z_pred))[1])

    plt.plot(n_iter, maxcs[-1], 'o', color='blue')
    plt.draw()
    plt.pause(0.05)

df = save_results(fname='./cca_res.csv', r=maxcs, N=N, method='CCA', dataset='lorenz')

plt.ioff()
plt.figure()
mngr = plt.get_current_fig_manager()
mngr.window.wm_geometry("+%d+%d" % (1000, 0))
plt.hist(maxcs)
plt.xlim(0, 1)
plt.show()
