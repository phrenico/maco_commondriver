"""Run AniSOM on the logmap data and comparison"""
import numpy as np

from tqdm import tqdm

import torch

from data_generators import LogmapExpRunner, time_delay_embedding, comp_ccorr, get_maxes, save_results, train_test_split

from cdriver.network.anisom import AniSOM
import matplotlib.pyplot as plt
import pandas as pd


def myfun(x, *args, **kwargs):
  return torch.linalg.eigh(x)

torch.symeig = myfun

if __name__ == "__main__":
    # 1. Generate data
    N = 50  # number of realizations
    n = 20_000  # Length of time series
    rint = (3.8, 4.)  # interval to chose from the value of r parameter
    A0 = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]])  # basic connection structure
    A = np.array([[1., 0., 0.],
                  [0.3, 1., 0.],
                  [0.4, 0., 1.]])
    train_split = 0.5

    dataset = [LogmapExpRunner(nvars=3, baseA=A0, r_interval=rint).gen_experiment(n=n, A=A, seed=i)[0] for i in
               tqdm(range(N))]

    save_data = False
    save_path = './dataset/'
    if save_data:
        _ = [pd.DataFrame(i).to_csv(save_path + 'logmap_{}.csv'.format(j)) for j, i in enumerate(dataset)]
        exit()

    params = [LogmapExpRunner(nvars=3, baseA=A0, r_interval=rint).gen_experiment(n=2, A=A, seed=i)[1] for i in range(N)]

    # Define parameters and layers for deep model
    d_embed = 3
    d_grid = 2
    d_space = d_embed
    sizes = [40, 20]

    maxcs = []
    maxcs2 = []
    maxcs3 = []

    for i in tqdm(range(N)):
        data = dataset[i]

        z = data[:-(d_embed - 1), 0]
        X = time_delay_embedding(data[:, 1], dimension=d_embed)
        Y = time_delay_embedding(data[:, 2], dimension=d_embed)

        X_train, Y_train, z_train, X_test, Y_test, z_test = train_test_split(X, Y, z, train_split)


        ani = AniSOM(space_dim=d_space, grid_dim=d_grid, sizes=sizes)

        ani.fit(torch.Tensor(X_train), torch.Tensor(Y_train), epochs=1, disable_tqdm=True)
        pred = ani.predict(torch.Tensor(X_test))

        tau, c = comp_ccorr(pred[:, 1], z_test)

        maxcs.append(get_maxes(tau, c)[1])


    # save out results
    df = save_results(fname='./anisom_res.csv', r=maxcs, N=N, method='ASOM', dataset='logmap_fixed')

    # 3. Plot results
    plt.figure()
    plt.hist(maxcs)
    plt.show()