import numpy as np
from matplotlib import pyplot as plt
# from sklearn.decomposition import PCA, FastICA
from sklearn.cross_decomposition import CCA

from sklearn.preprocessing import scale
import sys
sys.path.append('../')

from data_generators import time_delay_embedding, comp_ccorr, get_maxes
from scipy.signal import correlate, correlation_lags

data_path = '../../../data/lorenz.npz'
data = np.load(data_path)


T = 1000_000
ds = 200
X = data['v'][:T:ds, 3:6]
Y = data['v'][:T:ds, 6:]
x = data['v'][:T:ds, 1]
x2 = data['v'][:T:ds, 4]

cca = CCA(n_components=1, max_iter=1000)
cca.fit(X, Y)

zpred, zpred2 = cca.transform(X, Y)

# plt.plot(scale(zpred))
# plt.plot(scale(x[:T]))
# plt.plot(scale(x2[:T]))

z = zpred[:, 0]

lags = correlation_lags(T//ds, T//ds, 'full')
c = 1/T * correlate(scale(x), scale(z), 'full')
c = comp_ccorr(scale(x), scale(z))[1]
c2 = 1/T * correlate(scale(x2), scale(z), 'full')
c3 = 1/T * correlate(scale(x), scale(x2), 'full')
dt = 1e-3 * ds

plt.figure()
plt.plot(dt*lags, c, label='hcc, z', ls='-', lw=3.)
plt.plot(dt*lags, c2, label='x2, z')
plt.plot(dt*lags, c3, label='hcc, x2')

plt.axvline(0, color='k', ls='--')
plt.xlim(-10, 10)
# plt.ylim(0.25, 0.55)
plt.legend()
plt.show()