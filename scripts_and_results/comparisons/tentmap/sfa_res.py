'''Run experiments with SFA

'''
import numpy as np
import matplotlib.pyplot as plt

import sksfa

from tqdm import tqdm
from sklearn.preprocessing import scale

import sys
sys.path.append('../')
sys.path.append('../../../')

from data_generators import LogmapExpRunner, comp_ccorr, get_maxes, train_test_split, save_results, time_delay_embedding
from cdriver.datagen.tent_map import TentMapExpRunner

if __name__ == "__main__":
    plt.ion()
    plt.figure(figsize=(10, 10))
    plt.show()
    mngr = plt.get_current_fig_manager()
    mngr.window.wm_geometry("+%d+%d" % (0, 0))
    plt.xlim(-1, 100)
    plt.ylim(0, 1)

    N = 50
    n = 20_000
    train_split = 0.8
    valid_split = 0.1
    d_embed = 2

    aint = (2, 10.)  # interval to chose from the value of r parameter
    A0 = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]])  # basic connection structure

    dataset = [TentMapExpRunner(nvars=3, baseA=A0, a_interval=aint).gen_experiment(n=n, seed=i)[0] for i in
               tqdm(range(N))]

    d_embed = 3
    maxcs = []
    for n_iter in tqdm(range(N)):
        data = dataset[n_iter]

        z = data[:-(d_embed - 1), 0]
        X = time_delay_embedding(data[:, 1], dimension=d_embed)
        Y = time_delay_embedding(data[:, 2], dimension=d_embed)

        X_train, Y_train, z_train, X_test, Y_test, z_test = train_test_split(X, Y, z, train_split)
        D_train = np.concatenate([X_train, Y_train], axis=1)
        D_test = np.concatenate([X_test, Y_test], axis=1)

        # 2. Run SFA
        sfa = sksfa.SFA(n_components=1)
        sfa.fit(D_train)
        zpred = sfa.transform(D_test).squeeze()

        tau, c = comp_ccorr(zpred, z_test)
        maxcs.append(get_maxes(tau, c)[1])

        plt.plot(n_iter, maxcs[-1], 'o', color='blue')
        plt.draw()
        plt.pause(0.05)

    # Save results
    df = save_results(fname='./sfa_res.csv', r=maxcs, N=N, method='SFA', dataset='logmap_fixed')

    # 3. Plot results
    plt.figure()
    plt.hist(maxcs)
    plt.show()