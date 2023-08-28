import numpy as np
from matplotlib import pyplot as plt
# from sklearn.decomposition import PCA, FastICA
from sklearn.cross_decomposition import CCA

from sklearn.preprocessing import scale
import sys
sys.path.append('../')

from data_generators import time_delay_embedding, comp_ccorr, get_maxes, train_test_split
from scipy.signal import correlate, correlation_lags
import dca
DCA = dca.DynamicalComponentsAnalysis

data_path = '../../../data/lorenz.npz'
data = np.load(data_path)


T = 1000_000
ds = 200
X = data['v'][:T:ds, 3:6]
Y = data['v'][:T:ds, 6:]
x = data['v'][:T:ds, 1]
x2 = data['v'][:T:ds, 4]

train_split = 0.5
X_train, Y_train, z_train, X_test, Y_test, z_test = train_test_split(X, Y, x, train_split)

D_train = np.concatenate([X_train, Y_train], axis=1)
D_test = np.concatenate([X_test, Y_test], axis=1)

# 2. Run DCA
n_components = 1
dca_model = DCA(d=n_components, T=5, n_init=10)
dca_model.fit(D_train)
zpred = dca_model.transform(D_test)  # .squeeze()
print("sdfasdfs", zpred.shape)

z = zpred[:, 0]

lags = correlation_lags(T//ds*train_split, T//ds*train_split, 'full')

c = comp_ccorr(scale(z_test), scale(z))[1]
# c2 = 1/T * correlate(scale(x2), scale(z), 'full')
# c3 = 1/T * correlate(scale(x), scale(x2), 'full')
dt = 1e-3 * ds

plt.figure()
plt.plot(dt*lags, c, label='hcc, z', ls='-', lw=3.)
# plt.plot(dt*lags, c2, label='x2, z')
# plt.plot(dt*lags, c3, label='hcc, x2')

plt.axvline(0, color='k', ls='--')
plt.xlim(-10, 10)

plt.legend()
plt.show()