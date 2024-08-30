"""Run AniSOM on the logmap data and comparison"""
import numpy as np

from tqdm import tqdm

import torch

import sys
sys.path.append('../')
sys.path.append('../../../')


from cdriver.network.anisom import AniSOM
from cdriver.preprocessing.splitters import train_valid_test_split
from cdriver.preprocessing.tde import time_delay_embedding
from cdriver.savers.saver import  save_results
from cdriver.evaluate.evalz import comp_ccorr, get_maxes
from cdriver.datagen.tent_map import gen_tentmapdata

from scripts.datagen_scripts.datagen_config import tentmapgen_params
from config_tentmapres import train_split, interim_res_path, valid_split

import matplotlib.pyplot as plt


import matplotlib
matplotlib.use('TkAgg')




if __name__ == "__main__":
    # 0. Set up plotting
    plt.ion()
    plt.figure(figsize=(10, 10))
    mngr = plt.get_current_fig_manager()
    mngr.window.wm_geometry("+%d+%d" % (0, 0))
    plt.show()
    plt.xlim(-1, tentmapgen_params['N'])
    plt.ylim(0, 1)

    # 1. Generate data
    N = tentmapgen_params['N']  # number of realizations
    dataset, params = gen_tentmapdata(tentmapgen_params)


    # Define parameters and layers for deep model
    d_embed = 3
    d_grid = 2
    d_space = d_embed
    sizes = [40, 20]


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
        plt.pause(0.01)


    # save out results
    df = save_results(fname=interim_res_path / 'anisom_res.csv',
                      r=maxcs,
                      N=N,
                      method='ASOM',
                      dataset='tentmap')

    # 3. Plot results
    # plt.ioff()
    plt.figure()
    plt.hist(maxcs)
    plt.show()
    plt.close()