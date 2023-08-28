import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
sys.path.append('../')
from data_generators import LogmapExpRunner, comp_ccorr, get_maxes, train_test_split, save_results, time_delay_embedding
from shrec.models import RecurrenceManifold
from sklearn.preprocessing import scale
from scipy.signal import correlate, correlation_lags
# import torch

data_path = '../../../data/lorenz.npz'
data = np.load(data_path)


T = 1000_000
ds = 200
X = data['v'][:T:ds, 3:]
x = data['v'][:T:ds, 1]
x2 = data['v'][:T:ds, 4]

train_split = 0.5
X_train, Y_train, z_train, X_test, Y_test, z_test = train_test_split(X, X, x, train_split)

model = RecurrenceManifold(d_embed=1)
z = model.fit_predict(X_train)


lags = correlation_lags(T//ds*train_split, T//ds*train_split, 'full')
# c = 1/T * correlate(scale(x), scale(z), 'full')
c = comp_ccorr(scale(z_train), scale(z))[1]
# c2 = 1/T * correlate(scale(x2), scale(z), 'full')
# c3 = 1/T * correlate(scale(x), scale(x2), 'full')
dt = 1e-3 * ds

plt.figure()
plt.plot(dt*lags, c, label='hcc, z', ls='-', lw=3.)
# plt.plot(dt*lags, c2, label='x2, z')
# plt.plot(dt*lags, c3, label='hcc, x2')

plt.axvline(0, color='k', ls='--')
plt.xlim(-10, 10)
# plt.ylim(0.25, 0.55)
plt.legend()
plt.show()
