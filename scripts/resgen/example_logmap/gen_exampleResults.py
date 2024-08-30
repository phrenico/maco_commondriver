"""Generates the example results on the logistic map example
"""
import os

import numpy as np
from tqdm import tqdm
import sys
sys.path.append('/home/phrenico/Projects/Codes/maco_commondriver')

from cdriver.network.maco import MaCo
from cdriver.savers.saver import  save_results
from cdriver.evaluate.evalz import comp_ccorr, get_maxes
from cdriver.datagen.logmap import gen_logmapdata

import torch
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

from pathlib import Path
import pickle

from scripts.datagen_scripts.datagen_config import logmapgen_params
# from config_logmapres import interim_res_path, train_split

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

    train_dataset = TensorDataset(transforms.ToTensor()(x_train), transforms.ToTensor()(y_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    test_loader = transforms.ToTensor()(x_test), transforms.ToTensor()(y_test)
    valid_loader = transforms.ToTensor()(x_valid), transforms.ToTensor()(y_valid)
    return train_loader, test_loader, valid_loader, z_test


def main():
    # Parameters
    dx = 1
    dy = 2
    dz = 1
    nh = 20 # number of hidden units
    mapper_kwargs = dict(n_h1=nh, n_h2=nh)
    coach_kwargs = dict(n_h1=nh, n_out=1)
    preprocess_kwargs = dict(tau=1)
    device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    n_models = 10  # number of models to train

    trainset_size = 80
    testset_size = 10
    validset_size = 10

    n_epochs = 2_000
    batch_size = 1_000
    n = 2_000
    lr = 1e-2

    dataset, params = gen_logmapdata(logmapgen_params)
    data = dataset[0][:n]

    train_loader, test_loader, _, z_test = get_loaders(data,
                                                       batch_size=batch_size,
                                                       trainset_size=trainset_size,
                                                       testset_size=testset_size,
                                                       validset_size=validset_size)
    models = [MaCo(Ex=dx, Ey=dy, Ez=dz,
                   mh_kwargs=mapper_kwargs,
                   ch_kwargs=coach_kwargs,
                   preprocess_kwargs=preprocess_kwargs,
                   device=device) for i in range(n_models)]

    # Train models
    train_losses = []
    test_loss = []
    for i in tqdm(range(n_models), disable=False):
        train_losses += [models[i].train_loop(train_loader,
                                              n_epochs,
                                              lr=lr,
                                              disable_tqdm=True)]
        test_loss += [models[i].test_loop(test_loader)]
    train_losses = np.array(train_losses).T
    test_loss = np.array(test_loss).T

    # Pick the best model on the test set
    ind_best_model = np.argmin(test_loss)
    best_model = models[ind_best_model]

    valid_loss, x_pred, z_pred, hz_pred = best_model.valid_loop(test_loader)

    # print("shape of z_pred: {} and the shape of z_test: {}".format(z_pred.shape, z_test.shape))
    tau, c = comp_ccorr(z_pred, z_test)

    maxcs = [get_maxes(tau, c)[1], ]


    # compute the correlation between the hidden variables
    r_reconst = []
    r_predict = []
    for model in tqdm(models):
        preds = model.valid_loop(test_loader)
        print(preds[1].shape, preds[2].shape, test_loader[1].squeeze().shape)
        # exit()
        r_predict += [np.corrcoef(preds[1], test_loader[0].squeeze()[1:])[0, 1] ]
        r_reconst += [np.corrcoef(preds[2], z_test.squeeze()[:-1])[0, 1]]

    # Save out results
    res_dict = {'cc_pred': z_pred,
                'cc_valid': z_test[:-1],
                'x_valid': test_loader[0].squeeze().detach().numpy()[1:],
                'x_past_valid': test_loader[0].squeeze()[:-1],
                'x_pred': x_pred,
                'Y_1_valid': test_loader[1].squeeze()[:-1],
                'Y_2_valid': test_loader[1].squeeze()[1:],
                }
    for label, value in res_dict.items():
        print(label, value.shape)

    df = pd.DataFrame(res_dict)

    # Save out the Results (uncomment to rewrite the current results)
    respath = Path('/home/phrenico/Projects/Codes/maco_commondriver/results/final/example_logmap')
    os.makedirs(respath, exist_ok=True)
    df.to_csv(respath / 'mappercoach_res.csv')
    np.save(respath / 'learning_curves.npy', train_losses)
    np.save(respath / 'test_loss.npy', test_loss)
    torch.save(best_model, respath / 'best_model.pth')
    with open(respath / 'models.pkl', 'wb') as f:
        pickle.dump(models, f)
    pd.DataFrame({'r_predict':r_predict,
                  'r_reconst':r_reconst}).to_csv(respath / 'r_values.csv')
    print(np.corrcoef(z_test[:-1], z_pred))

if __name__ == "__main__":
    main()
