"""Run AniSOM on the logmap data and comparison"""
import numpy as np

from tqdm import tqdm

import torch

import sys
sys.path.append('../')
sys.path.append('../../../')

from data_generators import  comp_ccorr, get_maxes, train_valid_test_split, save_results, time_delay_embedding
from cdriver.datagen.tent_map import TentMapExpRunner
from cdriver.network.anisom import AniSOM
import matplotlib.pyplot as plt
import pandas as pd

from data_config import N, n, A0, aint

import matplotlib
matplotlib.use('TkAgg')




if __name__ == "__main__":
    # 0. Set up plotting
    plt.ion()
    plt.figure(figsize=(10, 10))
    mngr = plt.get_current_fig_manager()
    mngr.window.wm_geometry("+%d+%d" % (0, 0))
    plt.show()
    plt.xlim(-1, 100)
    plt.ylim(0, 1)

    # 1. Generate data
    dataset = [TentMapExpRunner(nvars=3, baseA=A0, a_interval=aint).gen_experiment(n=n, seed=i)[0] for i in
               tqdm(range(N))]



    # Define parameters and layers for deep model
    d_embed = 3
    d_grid = 2
    d_space = d_embed
    sizes = [40, 20]

    # Run the  Reconstructions on the Datasets
    train_split = 0.8
    valid_split = 0.1

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


        ani = AniSOM(space_dim=d_space, grid_dim=d_grid, sizes=sizes)

        ani.fit(torch.Tensor(X_train), torch.Tensor(Y_train), epochs=1, disable_tqdm=True)
        pred = ani.predict(torch.Tensor(X_test))

        tau, c = comp_ccorr(pred[:, 1], z_test)

        maxcs.append(get_maxes(tau, c)[1])
        plt.plot(n_iter, maxcs[-1], 'o', color='blue')
        plt.draw()
        plt.pause(0.05)


    # save out results
    df = save_results(fname='./anisom_res.csv', r=maxcs, N=N, method='ASOM', dataset='tentmap')

    # 3. Plot results
    plt.ioff()
    plt.figure()
    plt.hist(maxcs)
    plt.show()