import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import scale
import sys
sys.path.append('../')

from data_generators import time_delay_embedding, comp_ccorr, get_maxes, train_test_split, save_results
from scipy.signal import correlate, correlation_lags

N = 100
train_split = 0.5
maxcs = []
for i in range(N):
    data_path = '../../../data/lorenz/lorenz_{}.npz'.format(i)
    data = np.load(data_path)

    X = data['v'][:, 3:]
    z = data['v'][:, 1]
    T = X.shape[0]

    X_train, Y_train, z_train, X_test, Y_test, z_test = train_test_split(X, X, z, train_split)

    pca =  PCA(n_components=1).fit(X_train)
    zpred = pca.transform(X_test)[:, 0]

    maxcs.append(get_maxes(*comp_ccorr(z_test, zpred))[1])

df = save_results(fname='./pca_res.csv', r=maxcs, N=N, method='PCA', dataset='lorenz_fixed')

plt.hist(maxcs)
plt.xlim(0, 1)
plt.show()
