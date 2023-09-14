'''Script to test the effect of dataset size on the MACO algorithm
We Generate N=15 instances of coupled Logistic map systems, then we apply the algorithm on time series of different lengths.
steps are:
1. Generate N=15 instances of coupled Logistic map systems
2. Create chunked time series of different lengths
3. Train the MACO algorithm on each of the variable length time series
4. Plot results in the function of length of time series
'''
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns

import sys
sys.path.append("../../../")

from network.maco import MaCo
from scripts_and_results.comparisons.data_generators import LogmapExpRunner, comp_ccorr, get_maxes, save_results
import torch
import torchvision.transforms as transforms
from sklearn.preprocessing import scale

from functools import partial
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


def split_sets(x, y, z, trainset_size, testset_size, validset_size):
    """

    :param x: input data
    :param y: target dataq
    :param z: hidden variable
    :param trainset_size: training set size in percentage
    :param testset_size: test set size in percentage
    :param validset_size:   validation set size in percentage
    :return: splitted data into train, test and validation sets
    """
    n = x.shape[0]
    n_trainset = int(trainset_size * n / 100)
    n_testset = int(testset_size * n / 100)
    n_validset = int(validset_size * n / 100)

    x_trainset = x[:n_trainset]
    x_testset = x[n_trainset:n_trainset + n_testset]
    x_validset = x[n_trainset + n_testset:]

    y_trainset = y[:n_trainset]
    y_testset = y[n_trainset:n_trainset + n_testset]
    y_validset = y[n_trainset + n_testset:]

    z_trainset = z[:n_trainset]
    z_testset = z[n_trainset:n_trainset + n_testset]
    z_validset = z[n_trainset + n_testset:]
    return ((x_trainset, y_trainset, z_trainset),
            (x_testset, y_testset, z_testset),
            (x_validset, y_validset, z_validset))


def get_loaders(data, batch_size, trainset_size=50, testset_size=50, validset_size=0):
    """get data loaders for a dataset

    :param data:
    :param batch_size:
    :param trainset_size:
    :param testset_size:
    :param validset_size:
    :return:
    """


    x = data[:, 1:2]
    y = data[:, 2:3]
    z = data[:-1, 0]  # we only use it in the final evaluation of the learned represenation

    # Split into Traing test and validation sets
    splitted_data = split_sets(x, y, z, trainset_size, testset_size, validset_size)
    (x_train, y_train, z_train), (x_test, y_test, z_test), (x_valid, y_valid, z_valid) \
        = splitted_data

    # print("shapes in dataloader:", x_train.shape, y_train.shape, z_train.shape)
    train_dataset = TensorDataset(transforms.ToTensor()(x_train), transforms.ToTensor()(y_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    test_loader = transforms.ToTensor()(x_test), transforms.ToTensor()(y_test)
    valid_loader = transforms.ToTensor()(x_valid), transforms.ToTensor()(y_valid)
    return train_loader, test_loader, valid_loader, z_test

# 1. Generate random Logistic datasets
N = 15  # number of realizations
Ls = list(range(100, 1_000, 200)) + list(range(1_000, 3_001, 1_000))
n = max(Ls)  # Length of time series
rint = (3.8, 4.)  # interval to chose from the value of r parameter
A0 = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]])  # basic connection structure


train_split = 0.8
valid_split = 0.1
test_split = 0.1

datasets, params = zip(
    *[LogmapExpRunner(nvars=3, baseA=A0, r_interval=rint).gen_experiment(n=n, seed=i) for i in tqdm(range(N))])

n_epochs = 100
n_models = 10
dx = 1
dy = 2
dz = 1
nh = 20  # number of hidden units
mapper_kwargs = dict(n_h1=nh, n_h2=nh)
coach_kwargs = dict(n_h1=nh, n_out=1)
preprocess_kwargs = dict(tau=1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

maxdict = {}
for L in tqdm(Ls):
    maxcs = []
    for n_iter in tqdm(range(N)):
        data = datasets[n_iter].astype(float)[:L, :]

        # print("original data shape:", data.shape)

        train_loader, test_loader, _, z_test = get_loaders(data, batch_size=500, trainset_size=50,
                                                           testset_size=50, validset_size=0)




        models = [MaCo(Ex=dx, Ey=dy, Ez=dz,
                       mh_kwargs=mapper_kwargs,
                       ch_kwargs=coach_kwargs,
                       preprocess_kwargs=preprocess_kwargs,
                       device=device) for i in range(n_models)]



        # Train models
        train_losses = []
        test_loss = []
        for i in tqdm(range(n_models), disable=True):
            train_losses += [models[i].train_loop(train_loader, n_epochs, lr=1e-2, disable_tqdm=False)]
            test_loss += [models[i].test_loop(test_loader)]
        train_losses = np.array(train_losses).T

        # Pick the best model on the test set
        ind_best_model = np.argmin(test_loss)
        best_model = models[ind_best_model]

        valid_loss, x_pred, z_pred, hz_pred = best_model.valid_loop(test_loader)

        # print("shape of z_pred: {} and the shape of z_test: {}".format(z_pred.shape, z_test.shape))
        tau, c = comp_ccorr(z_pred, z_test)
        maxcs.append(get_maxes(tau, c)[1])
    maxdict[L] = maxcs.copy()

def create_df(maxdict):
    """Create a dataframe from a dictionary of lists

    :param maxdict: dictionary of lists
    :return: dataframe
    """
    keys = maxdict.keys()
    dfs = [pd.DataFrame(np.array([maxdict[key], len(maxdict[key]) * [key] ]).T, columns=['r2', 'L']) for key in keys]
    df = pd.concat(dfs, axis=0)
    return df

df = create_df(maxdict)
df.to_csv('./length_maco_res.csv')


sns.lineplot(x='L', y='r2', data=df)

plt.ylim(0, 1)
plt.show()