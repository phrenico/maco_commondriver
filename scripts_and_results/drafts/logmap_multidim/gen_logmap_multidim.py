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

from cdriver.network.maco import MaCo
from cdriver.visuals.respics import learnforcast, pred_rec, reconstruction
from scipy.stats import norm



def load_data(batch_size, trainset_size, testset_size, validset_size, data_fn):
    data = pd.read_csv(data_fn, index_col=0).values
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5), (0.3)),
                                    torch.Tensor.float,
                                    partial(torch.squeeze, axis=0)])

    x = transform(data[:, 2:3])
    y = transform(data[:, 3:4])
    # z = scale(np.array([data[1:-2, 1], data[2:-1, 1]]).T)  # we only use it in the final evaluation of the learned representaion
    z = scale(data[2:-1, :2])

    Q = torch.cat((x[2:-1], y[:-3], y[1:-2], y[2:-1]), axis=-1)
    Targ = x[3:]

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
    dy = 3
    dz = 2
    nh = 20 # number of hidden units
    mapper_kwargs = dict(n_h1=nh, n_h2=nh)
    coach_kwargs = dict(n_h1=nh)
    device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    n_models = 10  # number of models to train

    trainset_size = 60
    testset_size = 20
    validset_size = 20
    c = 1e-1 # regularization parameter

    n_epochs = 4000
    batch_size = 2000
    data_fn = '../../../data/logmap_2d_data2.csv'

    # Load in data & preprocessing to pytorch
    train_loader, test_loader, valid_loader, z_valid = load_data(batch_size, trainset_size,
                                                                 testset_size, validset_size,
                                                                 data_fn)

    models = [MaCo(Ex=dx, Ey=dy, Ez=dz, c=c,
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
        print(preds[2][:, -1].shape, preds[1].shape)
        r_reconst += [np.corrcoef(preds[2][:, -1], z_valid[:, 1])[0, 1]]

    # Save out results
    res_dict = {'x_valid': valid_loader[1].squeeze().detach().numpy(),
                'x_past_valid': valid_loader[0][:, 0],
                'x_pred': x_pred,
                'Y_1_valid': valid_loader[0][:, 1],
                'Y_2_valid': valid_loader[0][:, 2],
                }
    for i in range(dz):
        res_dict['cc_{}_pred'.format(i+1)] = z_pred[:, i]
        res_dict['cc_{}_valid'.format(i + 1)] = z_valid[:, i]
    [print(i, res_dict[i].shape) for i in res_dict]
    df = pd.DataFrame(res_dict)
    rdf = pd.DataFrame({'r_predict':r_predict, 'r_reconst':r_reconst})

    # Save out the Results (uncomment to rewrite the current results)
    save_path = './logmap_2d_res/'
    df.to_csv(save_path+'mappercoach_res.csv')
    np.save(save_path+'/learning_curves.npy', train_losses)
    np.save(save_path+'test_loss.npy', test_loss)
    torch.save(best_model, save_path+'best_model.pth')
    with open(save_path+'/models.pkl', 'wb') as f:
        pickle.dump(models, f)
    rdf.to_csv(save_path+'r_values.csv')
    print(np.corrcoef(z_valid[:, 1], z_pred[:, -1]), np.corrcoef(z_valid[:, 1], z_pred[:, 0]))

    fig1 = learnforcast(df, train_losses, test_loss)
    # fig2 = reconstruction(df)
    # fig3 = pred_rec(rdf)
    # figs = [fig1, fig2, fig3]
    #
    # [figs[i].savefig(save_path+'fig{}.png'.format(i+1), dpi=300) for i in range(len(figs))]


    from cdriver.evaluate.evalz import eval_lin

    regmod, regmod2 = eval_lin(z_valid, z_pred)
    M = regmod.coef_
    M2 = regmod2.coef_
    # print(np.linalg.pinv(M), M2)

    print(z_valid.shape, z_pred.shape)

    Y_pred = regmod.predict(z_valid)
    C = np.corrcoef(z_valid, Y_pred, rowvar=False)
    print(C)

    print(np.corrcoef(z_valid, z_pred, rowvar=False))
    plt.show()

if __name__ == "__main__":
    main()