"""Generates the example results on the logistic map example
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms

from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

import torch
from torch.nn import Module, Sequential, Linear, ReLU, MSELoss, PReLU
from torch import Tensor
import torchvision.transforms as transforms
from torch.nn.functional import relu
import torch.optim as optim
from torch.utils.data import random_split

from tqdm import tqdm

from sklearn.preprocessing import scale
from scipy.signal import correlate, correlation_lags
from scipy.stats import rankdata
from functools import partial


# @todo: fix the device to GPU-compatible test it
# @todo: fix the parameter settings

def get_mapper(n_in, n_h1, n_h2, n_out):
    mapper = Sequential(Linear(n_in, n_h1),
                        ReLU(),
                        Linear(n_h1, n_h2),
                        ReLU(),
                        Linear(n_h2, n_out))
    return mapper


def get_coach(n_in, n_h1, n_out):
    coach = Sequential(Linear(n_in, n_h1),
                       ReLU(),
                       Linear(n_h1, n_out))
    return coach


class MaCo(torch.nn.Module):
    def __init__(self, Ex, Ey, Ez, mh_kwargs, ch_kwargs, device):
        super().__init__()
        self.mapper = get_mapper(n_in=Ey, n_out=Ez, **mh_kwargs)
        self.coach_x = get_coach(Ex + Ez, n_out=1, **ch_kwargs)

        self.Ex = Ex
        self.Ey = Ey
        self.Ez = Ez
        self.mh_params = mh_kwargs
        self.ch_params = ch_kwargs
        self.train_loss_history = []
        self.criterion = MSELoss()
        self.device = device

    def forward(self, q):
        z = self.mapper.forward(q[:, self.Ex:self.Ex + self.Ey])
        hz = relu(z)

        mx = torch.cat((hz, q[:, :self.Ex]), axis=1)

        pred = self.coach_x.forward(mx)
        return pred, z, hz

    def train_loop(self, loader, n_epochs, lr=1e-2):
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        loss_hist = self.train_loss_history.copy()
        for epoch in tqdm(range(n_epochs), leave=False):
            losses = []
            for i, batch in enumerate(loader):
                q, target = batch

                self.optimizer.zero_grad()

                pred, z, hz = self.forward(q.to(self.device))

                loss = self.criterion(target.to(self.device), pred)

                loss.backward()
                self.optimizer.step()

                losses.append(loss.item())
            loss_hist.append(np.mean(losses))
        self.train_loss_history = loss_hist.copy()
        return loss_hist

    def test_loop(self, loader):
        q, target = loader
        pred, z, hz = self.forward(q)
        loss = self.criterion(target, pred).item()
        return loss

    def valid_loop(self, loader):
        q, target = loader
        pred, z, hz = self.forward(q)
        loss = self.criterion(target, pred).item()
        return loss, pred.squeeze().detach().numpy(), z.squeeze().detach().numpy(), hz.squeeze().detach().numpy()

def load_data(batch_size, trainset_size, testset_size, validset_size):
    data = pd.read_csv('../data/sampledata.csv', index_col=0).values
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5), (0.3)),
                                    torch.Tensor.float,
                                    partial(torch.squeeze, axis=0)])

    x = transform(data[:, 1:2])
    y = transform(data[:, 2:3])
    z = scale(data[:-1, 0])  # we only use it in the final evaluation of the learned represenation
    #
    Q = torch.cat((x[:-1], y[:-1], y[1:]), axis=-1)
    Targ = x[1:]

    # Split into Traing test and validation sets
    splitted_data = split_sets([Q, Targ, z],
                               trainset_size,
                               testset_size,
                               validset_size)
    (Q_train, Targ_train, z_train), (Q_test, Targ_test, z_test), (Q_valid, Targ_valid, z_valid) \
        = splitted_data

    train_loader = make_batches(Q_train, Targ_train, batch_size=batch_size)
    test_loader = Q_test, Targ_test
    valid_loader = Q_valid, Targ_valid
    return train_loader, test_loader, valid_loader, z_valid

def make_batches(Q, target, batch_size):
    maxsize = Q.shape[0]
    i = np.arange(maxsize)
    np.random.shuffle(i)
    return list(zip(Q[i, :].split(batch_size), target[i, :].split(batch_size)))

def split_sets(data, trainset_size, testset_size, validset_size):
    def split_one(X, trainset_size, testset_size, validset_size):
        N = X.shape[0]
        X_train = X[:int(trainset_size * N)]
        X_test = X[int(trainset_size * N):int((trainset_size + testset_size) * N)]
        X_valid = X[int((trainset_size + testset_size) * N):]

        return X_train, X_test, X_valid

    # normalize sizes
    S = trainset_size + testset_size + validset_size
    trainset_part, testset_part, validset_part = (i / S for i in [trainset_size, testset_size, validset_size])

    splitteds = [split_one(i, trainset_part, testset_part, validset_part) for i in data]

    return list(zip(*splitteds))


def main():
    # Parameters
    dx = 1
    dy = 2
    dz = 1
    nh = 20 # number of hidden units
    mapper_kwargs = dict(n_h1=nh, n_h2=nh)
    coach_kwargs = dict(n_h1=nh)
    device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    n_models = 20  # number of models to train

    trainset_size = 80
    testset_size = 10
    validset_size = 10

    n_epochs = 4000
    batch_size = 2000

    # Load in data & preprocessing to pytorch
    train_loader, test_loader, valid_loader, z_valid = load_data(batch_size, trainset_size,
                                                                 testset_size, validset_size)

    models = [MaCo(Ex=dx, Ey=dy, Ez=dz,
                   mh_kwargs=mapper_kwargs, ch_kwargs=coach_kwargs, device=device) for i in range(n_models)]

    # Train models
    train_losses = []
    test_loss = []
    for i in tqdm(range(n_models)):
        train_losses += [models[i].train_loop(train_loader, n_epochs, lr=1e-2)]
        test_loss += [models[i].test_loop(test_loader)]
    train_losses = np.array(train_losses).T

    # Pick the best model on the test set
    ind_best_model = np.argmin(test_loss)
    best_model = models[ind_best_model]

    # compute reconstruction on the validation set
    valid_loss, x_pred, z_pred, hz_pred = best_model.valid_loop(valid_loader)

    # Compute correlation for all models on validation set
    r_reconst = []
    r_predict = []
    for model in tqdm(models):
        preds = model.valid_loop(valid_loader)
        r_predict += [np.corrcoef(preds[1], valid_loader[1][:, 0])[0, 1] ]
        r_reconst += [np.corrcoef(preds[2], z_valid)[0, 1]]

    # Save out results
    res_dict = {'cc_pred': z_pred,
                'cc_valid': z_valid,
                'x_valid': valid_loader[1].squeeze().detach().numpy(),
                'x_past_valid': valid_loader[0][:, 0],
                'x_pred': x_pred,
                'Y_1_valid': valid_loader[0][:, 1],
                'Y_2_valid': valid_loader[0][:, 2],
                }

    df = pd.DataFrame(res_dict)

    df.to_csv('./resdata/mappercoach_res.csv')
    np.save('./resdata/learning_curves.npy', train_losses)
    np.save('./resdata/test_loss.npy', test_loss)
    torch.save(best_model, './resdata/best_model.pth')
    with open('./resdata/models.pkl', 'wb') as f:
        pickle.dump(models, f)
    pd.DataFrame({'r_predict':r_predict, 'r_reconst':r_reconst}).to_csv('./resdata/r_values.csv')
    print(np.corrcoef(z_valid, z_pred))

if __name__ == "__main__":
    main()
