"""It's not running, use Google Colab in stead"""
import numpy as np

from tqdm import tqdm

from mvlearn.embed import DCCA
import torch

from data_generators import LogmapExpRunner, time_delay_embedding, comp_ccorr, get_maxes, save_results, train_test_split
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
    params = [LogmapExpRunner(nvars=3, baseA=A0, r_interval=rint).gen_experiment(n=2, A=A, seed=i)[1] for i in range(N)]

    # Define parameters and layers for deep model
    d_embed = 3
    features1 = d_embed  # Feature sizes
    features2 = d_embed
    layers1 = [256, 256, 1]  # nodes in each hidden layer and the output size
    layers2 = layers1.copy()

    maxcs = []
    maxcs2 = []
    maxcs3 = []

    for i in tqdm(range(N)):
        data = dataset[i]

        z = data[:-(d_embed - 1), 0]
        X = time_delay_embedding(data[:, 1], dimension=d_embed)
        Y = time_delay_embedding(data[:, 2], dimension=d_embed)

        X_train, Y_train, z_train, X_test, Y_test, z_test = train_test_split(X, Y, z, train_split)


        dcca = DCCA(input_size1=features1, input_size2=features2, n_components=1,
                    layer_sizes1=layers1, layer_sizes2=layers2, epoch_num=500,
                    use_all_singular_values=True)
        dcca.fit([X_train, Y_train])
        Xs_transformed = dcca.transform([X_test, Y_test])

        zp1, zp2 = Xs_transformed

        # tau, c = comp_ccorr(zp1[:, 0], z_test)
        # tau, c2 = comp_ccorr(zp2[:, 0], z_test)
        tau, c3 = comp_ccorr((zp1[:, 0] + zp2[:, 0]) / 2, z_test)

        # maxcs.append(get_maxes(tau, c)[1])
        # maxcs2.append(get_maxes(tau, c2)[1])
        maxcs3.append(get_maxes(tau, c3)[1])

    # save out results
    df = save_results(fname='./dcca_res.csv', r=maxcs3, N=N, method='DCCA', dataset='logmap_fixed')