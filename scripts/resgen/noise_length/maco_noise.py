'''Script to test the effect of dataset size on the MACO algorithm
We Generate N=15 instances of coupled Logistic map systems, then we apply the algorithm on time series of different lengths.
steps are:
1. Generate N=15 instances of coupled Logistic map systems
2. Create chunked time series of different lengths
3. Train the MACO algorithm on each of the variable length time series
4. Plot results in the function of length of time series
'''
from types import SimpleNamespace

import numpy as np
import pandas as pd
from tqdm import tqdm
import os

import sys
sys.path.append("../../../")

from cdriver.datagen.logmap import LogmapExpRunner
from cdriver.evaluate.evalz import comp_ccorr, get_maxes
from cdriver.network.maco import MaCo

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset

from config_noise_length import noise_params, final_res_path
from types import SimpleNamespace

def create_df(maxdict):
    """Create a dataframe from a dictionary of lists

    :param maxdict: dictionary of lists
    :return: dataframe
    """
    keys = maxdict.keys()
    dfs = [pd.DataFrame(np.array([maxdict[key], len(maxdict[key]) * [key] ]).T, columns=['r2', 'L']) for key in keys]
    df = pd.concat(dfs, axis=0)
    return df


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

p = SimpleNamespace(**noise_params)
os.makedirs(final_res_path, exist_ok=True)

# 1. Generate random Logistic datasets
datasets, params = zip(
    *[LogmapExpRunner(nvars=p.nvars,
                      baseA=p.A0,
                      r_interval=p.rint).gen_experiment(n=p.n,
                                                        seed=i) for i in tqdm(range(p.N))])


print("Data sigma: ",np.mean([i.std() for i in datasets]))
exit()

mapper_kwargs = dict(n_h1=p.nh, n_h2=p.nh)
coach_kwargs = dict(n_h1=p.nh, n_out=1)
preprocess_kwargs = dict(tau=p.tau)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

maxdict = {}
for L in tqdm(p.Ls, desc='Noise levels'):
    maxcs = []
    for n_iter in tqdm(range(p.N), desc='instances', leave=False):
        data = datasets[n_iter].astype(float)
        data[:, 1:] = data[:, 1:] + np.random.normal(loc=0, scale=L, size=data[:, 1:].shape)  # add observation noise to the observed time series

        train_loader, test_loader, valid_loader, z_test = get_loaders(data,
                                                           batch_size=p.batch_size,
                                                           trainset_size=p.trainset_size,
                                                           testset_size=p.testset_size,
                                                           validset_size=p.validset_size)
        models = [MaCo(Ex=p.dx, Ey=p.dy, Ez=p.dz,
                       mh_kwargs=mapper_kwargs,
                       ch_kwargs=coach_kwargs,
                       preprocess_kwargs=preprocess_kwargs,
                       device=device) for i in range(p.n_models)]

        # Train models
        train_losses = []
        valid_loss = []
        for i in tqdm(range(p.n_models),
                      disable=True,
                      desc='Models',
                      leave=False):
            train_losses += [models[i].train_loop(train_loader,
                                                  p.n_epochs,
                                                  lr=p.lr,
                                                  disable_tqdm=True)]
            valid_loss += [models[i].test_loop(valid_loader)]
        train_losses = np.array(train_losses).T

        # Pick the best model on the test set
        ind_best_model = np.argmin(valid_loss)
        best_model = models[ind_best_model]

        valid_loss, x_pred, z_pred, hz_pred = best_model.valid_loop(test_loader)

        # print("shape of z_pred: {} and the shape of z_test: {}".format(z_pred.shape, z_test.shape))
        tau, c = comp_ccorr(z_pred, z_test)
        maxcs.append(get_maxes(tau, c)[1])
    maxdict[L] = maxcs.copy()

df = create_df(maxdict)
df.to_csv(final_res_path / './noise_maco_res.csv')