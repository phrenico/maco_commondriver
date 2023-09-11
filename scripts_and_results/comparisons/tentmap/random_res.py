import numpy as np
import sys
sys.path.append('../')
from data_generators import comp_ccorr, get_maxes, save_results, train_valid_test_split, shuffle_phase, time_delay_embedding
from cdriver.datagen.tent_map import TentMapExpRunner

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

N = 50
n = 20_000
train_split = 0.8
valid_split = 0.1
d_embed = 3

aint = (2, 10.)  # interval to chose from the value of r parameter
A0 = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]])  # basic connection structure

dataset = [TentMapExpRunner(nvars=3, baseA=A0, a_interval=aint).gen_experiment(n=n, seed=i)[0] for i in
           tqdm(range(N))]
maxcs = []
for n_iter in tqdm(range(N)):
    data = dataset[n_iter]

    z = data[:-(d_embed - 1), 0]
    X = time_delay_embedding(data[:, 1], dimension=d_embed)
    Y = time_delay_embedding(data[:, 2], dimension=d_embed)

    X_train, Y_train, z_train, X_valid, Y_valid, z_valid, X_test, Y_test, z_test = train_valid_test_split(X, Y, z,
                                                                                                          train_split,
                                                                                                          valid_split)
    z_pred = shuffle_phase(z_test)

    maxcs.append(get_maxes(*comp_ccorr(z_test, z_pred))[1])

    plt.plot(n_iter, maxcs[-1], 'o', color='blue')
    plt.draw()
    plt.pause(0.05)

df = save_results(fname='./random_res.csv', r=maxcs, N=N, method='Random', dataset='tentmap')

plt.ioff()
plt.figure()
mngr = plt.get_current_fig_manager()
mngr.window.wm_geometry("+%d+%d" % (1000, 0))
plt.hist(maxcs)
plt.xlim(0, 1)
plt.show()