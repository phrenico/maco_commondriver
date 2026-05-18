'''Apply MACO to the Lorenz system and plot the results.

'''
import argparse
from pathlib import Path

import torch
import torchvision.transforms as transforms

from functools import partial
import os
import numpy as np
from cdriver.savers.saver import save_results
from cdriver.evaluate.evalz import comp_ccorr, get_maxes
from cdriver.network.maco import MaCo
from scripts.experiments.maco_utils import build_loaders, get_default_device, train_and_select_best_model
from scripts.experiments.config_loader import get_config, resolve_paths
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[3]


def typer(x, dtype=torch.float32):
    """Set the type of a tensor."""
    return x.type(dtype)


def myscaler(x, axis):
    return (x - x.mean(axis=axis, keepdim=True)) / x.std(axis=axis, keepdim=True)


def preprocess(X, Y):
    """Preprocess data."""
    global device

    common_transform = transforms.Compose([torch.tensor,
                                           torch.Tensor.float,
                                        #    scale,
                                           partial(torch.squeeze, axis=0)])

    X_target = common_transform(X[1:, :1])
    X_basic = common_transform(X[:-1])
    # Y_basic = common_transform(Y[:-1])
    Y_basic = common_transform(Y[1:])

    return X_basic, X_target, Y_basic


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None)
    args = parser.parse_args()
    cfg = resolve_paths(get_config('lorenz', args.config), _REPO_ROOT)

    N = cfg['data']['N']
    data_path_template = str(_REPO_ROOT / cfg['data']['data_path_template'])
    train_split = cfg['preprocessing']['train_split']
    valid_split = cfg['preprocessing']['valid_split']
    interim_res_path = cfg['paths']['interim_res_path']
    os.makedirs(interim_res_path, exist_ok=True)

    n_epochs = cfg['maco']['n_epochs']
    n_models = cfg['maco']['n_models']
    dx = cfg['maco']['dx']
    dy = cfg['maco']['dy']
    dz = cfg['maco']['dz']
    nh = cfg['maco']['n_hidden']
    lr = cfg['maco']['lr']
    batch_size = cfg['maco']['batch_size']
    tau = cfg['maco']['tau']

    device = get_default_device()
    mapper_kwargs = dict(n_h1=nh, n_h2=nh)
    coach_kwargs = dict(n_h1=nh, n_out=1)
    preprocess_kwargs = dict(tau=tau)
    loader_transform = transforms.Compose([transforms.ToTensor(), torch.squeeze])

    maxcs = []
    for n_iter in tqdm(range(N)):
        data_path = data_path_template.format(n_iter)
        data = np.load(data_path)

        # get variables of interest
        X = data['v'][:, 3:6]
        Y = data['v'][:, 6:]
        z = data['v'][:, 1]

        train_loader, test_loader, valid_loader, z_test = build_loaders(X,
                                                                        Y,
                                                                        z,
                                                                        batch_size=batch_size,
                                                                        trainset_size=int(100 * train_split),
                                                                        testset_size=100 - (train_split + valid_split) * 100,
                                                                        validset_size=valid_split * 100,
                                                                        transform=loader_transform)
        model_factory = lambda: MaCo(Ex=dx, Ey=dy, Ez=dz,
                                     mh_kwargs=mapper_kwargs,
                                     ch_kwargs=coach_kwargs,
                                     preprocess_kwargs=preprocess_kwargs,
                                     device=device)
        configure_model = lambda model: setattr(model, 'preprocess', preprocess)
        models, train_losses, valid_loss, best_model = train_and_select_best_model(model_factory,
                                                                                   train_loader,
                                                                                   valid_loader,
                                                                                   n_models,
                                                                                   n_epochs,
                                                                                   lr,
                                                                                   disable_tqdm=False,
                                                                                   leave=False,
                                                                                   desc='Training models',
                                                                                   configure_model=configure_model)
        valid_loss, x_pred, z_pred, hz_pred = best_model.valid_loop(test_loader)

        maxcs.append(get_maxes(*comp_ccorr(z_test, z_pred))[1])

    df = save_results(fname=interim_res_path / 'maco_res.csv',
                      r=maxcs,
                      N=N,
                      method='MaCo',
                      dataset='lorenz')

def typer(x, dtype=torch.float32):
    """Set the type of a tensor.

    :param x: tensor
    :param dtype: data type
    :return: tensor with the desired data type
    """
    return x.type(dtype)

def myscaler(x, axis):
    return (x - x.mean(axis=axis, keepdim=True)) / x.std(axis=axis, keepdim=True)


def preprocess(X, Y):
    """Preprocess data.

    :param X: input data
    :param Y: target data
    :return: preprocessed data
    """
    # print("Y shape in preprocessing", Y.shape)
    global device

    common_transform = transforms.Compose([torch.tensor, 
                                           torch.Tensor.float,
                                        #    scale,
                                           partial(torch.squeeze, axis=0)])

    X_target = common_transform(X[1:, :1])
    X_basic = common_transform(X[:-1])
    # Y_basic = common_transform(Y[:-1])
    Y_basic = common_transform(Y[1:])
    # print("Y shape after preprocessing", Y_basic.shape, Y_basic.dtype)
    
    return X_basic, X_target, Y_basic



device = get_default_device()
# apply MaCo
n_epochs = 200
n_models = 10
dx = 3
dy = 3
dz = 1
nh = 20  # number of hidden units
mapper_kwargs = dict(n_h1=nh, n_h2=nh)
coach_kwargs = dict(n_h1=nh, n_out=1)
preprocess_kwargs = dict(tau=1)
lr = 1e-2
batch_size = 1_000
loader_transform = transforms.Compose([transforms.ToTensor(), torch.squeeze])


maxcs = []
for n_iter in tqdm(range(N)):
    data_path = data_path_template.format(n_iter)
    data = np.load(data_path)


    # get variables of interest

    X = data['v'][:, 3:6]
    Y = data['v'][:, 6:]
    z = data['v'][:, 1]


    train_loader, test_loader, valid_loader, z_test = build_loaders(X,
                                                                    Y,
                                                                    z,
                                                                    batch_size=batch_size,
                                                                    trainset_size=int(100 * train_split),
                                                                    testset_size=100 - (train_split + valid_split) * 100,
                                                                    validset_size=valid_split * 100,
                                                                    transform=loader_transform)
    model_factory = lambda: MaCo(Ex=dx, Ey=dy, Ez=dz,
                                 mh_kwargs=mapper_kwargs,
                                 ch_kwargs=coach_kwargs,
                                 preprocess_kwargs=preprocess_kwargs,
                                 device=device)
    configure_model = lambda model: setattr(model, 'preprocess', preprocess)
    models, train_losses, valid_loss, best_model = train_and_select_best_model(model_factory,
                                                                               train_loader,
                                                                               valid_loader,
                                                                               n_models,
                                                                               n_epochs,
                                                                               lr,
                                                                               disable_tqdm=False,
                                                                               leave=False,
                                                                               desc='Training models',
                                                                               configure_model=configure_model)
    #
    valid_loss, x_pred, z_pred, hz_pred = best_model.valid_loop(test_loader)


    maxcs.append(get_maxes(*comp_ccorr(z_test, z_pred))[1])

df = save_results(fname=interim_res_path / './maco_res.csv',
                  r=maxcs,
                  N=N,
                  method='MaCo',
                  dataset='lorenz')