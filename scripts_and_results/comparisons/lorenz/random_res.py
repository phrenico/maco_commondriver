import numpy as np
import sys
sys.path.append('../')
from data_generators import comp_ccorr, get_maxes, save_results, train_test_split, shuffle_phase
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

plt.ion()
plt.figure(figsize=(10, 10))
mngr = plt.get_current_fig_manager()
mngr.window.wm_geometry("+%d+%d" % (0, 0))
plt.show()

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
    z_pred = shuffle_phase(z_test)

    maxcs.append(get_maxes(*comp_ccorr(z_test, z_pred))[1])

    plt.plot(n_iter, maxcs[-1], 'o', color='blue')
    plt.draw()
    plt.pause(0.05)

df = save_results(fname='./random_res.csv', r=maxcs, N=N, method='Random', dataset='lorenz')

plt.ioff()
plt.figure()
mngr = plt.get_current_fig_manager()
mngr.window.wm_geometry("+%d+%d" % (1000, 0))
plt.hist(maxcs)
plt.xlim(0, 1)
plt.show()