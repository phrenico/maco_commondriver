import numpy as np

from sklearn.cross_decomposition import CCA
import sys
sys.path.append('../')
from data_generators import  time_delay_embedding, comp_ccorr, get_maxes, save_results, train_test_split, train_valid_test_split
from cdriver.datagen.tent_map import TentMapExpRunner
import matplotlib.pyplot as plt

from tqdm import tqdm
import pandas as pd

if __name__=="__main__":


    # 1. Generate data
    N = 50  # number of realizations
    n = 20_000  # Length of time series
    aint = (2, 10.)  # interval to chose from the value of r parameter
    A0 = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]])  # basic connection structure

    plt.ion()
    plt.figure(figsize=(10, 10))
    plt.show()
    mngr = plt.get_current_fig_manager()
    mngr.window.wm_geometry("+%d+%d" % (0, 0))

    plt.xlim(-1, N+1)
    plt.ylim(0, 1)

    train_split = 0.8
    valid_split = 0.1
    d_embed = 3

    dataset = [TentMapExpRunner(nvars=3, baseA=A0, a_interval=aint).gen_experiment(n=n, seed=i)[0] for i in
                tqdm(range(N))]



    maxcs = []
    maxcs2 = []
    maxcs3 = []
    for n_iter in tqdm(range(N)):
        data = dataset[n_iter]

        z = data[:-(d_embed - 1), 0]
        X = time_delay_embedding(data[:, 1], dimension=d_embed)
        Y = time_delay_embedding(data[:, 2], dimension=d_embed)

        X_train, Y_train, z_train, X_valid, Y_valid, z_valid, X_test, Y_test, z_test = train_valid_test_split(X, Y, z,
                                                                                                        train_split,
                                                                                                        valid_split)

        cca = CCA(n_components=1)
        cca.fit(X_train, Y_train)

        zpred, zpred2 = cca.transform(X_test, Y_test)

        tau, c = comp_ccorr(zpred[:, 0], z_test)
        tau, c2 = comp_ccorr(zpred2[:, 0], z_test)
        tau, c3 = comp_ccorr((zpred[:, 0] + zpred2[:, 0]) / 2, z_test)

        maxcs.append(get_maxes(tau, c)[1])
        maxcs2.append(get_maxes(tau, c2)[1])
        maxcs3.append(get_maxes(tau, c3)[1])

        plt.plot(n_iter, maxcs3[-1], 'o', color='blue')
        plt.draw()
        plt.pause(0.05)

# Save results
df = save_results(fname='./cca_res.csv', r=maxcs3, N=N, method='CCA', dataset='tentmap')

plt.ioff()
plt.figure()
plt.hist(maxcs)
plt.hist(maxcs2)
plt.hist(maxcs3)
plt.show()