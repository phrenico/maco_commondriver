'''Apply MACO to the Lorenz system and plot the results.

'''
import numpy as np
from tqdm import tqdm

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

def get_loaders(X, Y, z, batch_size, trainset_size=50, testset_size=50, validset_size=0):
    """get data loaders for a dataset

    :param data:
    :param batch_size:
    :param trainset_size:
    :param testset_size:
    :param validset_size:
    :return:
    """


    # Split into Traing test and validation sets
    splitted_data = split_sets(X, Y, z, trainset_size, testset_size, validset_size)
    (x_train, y_train, z_train), (x_test, y_test, z_test), (x_valid, y_valid, z_valid) \
        = splitted_data

    # print("shapes in dataloader:", x_train.shape, y_train.shape, z_train.shape)
    common_transform = transforms.Compose([transforms.ToTensor(), torch.squeeze, ])
    train_dataset = TensorDataset(common_transform(x_train), common_transform(y_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    test_loader = common_transform(x_test), common_transform(y_test)
    valid_loader = common_transform(x_valid), common_transform(y_valid)
    return train_loader, test_loader, valid_loader, z_test

def typer(x, dtype=torch.float32):
    """Set the type of a tensor.

    :param x: tensor
    :param dtype: data type
    :return: tensor with the desired data type
    """
    return x.type(dtype)
def preprocess(X, Y):
    """Preprocess data.

    :param X: input data
    :param Y: target data
    :return: preprocessed data
    """
    # print("Y shape in preprocessing", Y.shape)

    common_transform = transforms.Compose([scale, transforms.ToTensor(), torch.squeeze, typer])

    X_target = common_transform(X[1:])
    X_basic = common_transform(X[:-1])
    Y_basic = common_transform(Y[:-1])
    # print("Y shape after preprocessing", Y_basic.shape, Y_basic.dtype)
    return X_basic, X_target, Y_basic



# apply MaCo
n_epochs = 100
n_models = 20
dx = 3
dy = 3
dz = 1
nh = 20  # number of hidden units
mapper_kwargs = dict(n_h1=nh, n_h2=nh)
coach_kwargs = dict(n_h1=nh, n_out=dx)
preprocess_kwargs = dict(tau=1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


plt.ion()
plt.figure(figsize=(10, 10))
plt.show()
mngr = plt.get_current_fig_manager()
mngr.window.wm_geometry("+%d+%d" % (0, 0))
plt.xlim(-1, 100)
plt.ylim(0, 1)

N = 100
train_split = 0.5
maxcs = []
for n_iter in tqdm(range(N)):
    data_path = '../../../data/lorenz/lorenz_{}.npz'.format(n_iter)
    data = np.load(data_path)


    # get variables of interest

    X = data['v'][:, 3:6]
    Y = data['v'][:, 6:]
    z = data['v'][:, 1]


    train_loader, test_loader, _, z_test = get_loaders(X, Y, z, batch_size=1000, trainset_size=50,
                                                       testset_size=50, validset_size=0)



    models = [MaCo(Ex=dx, Ey=dy, Ez=dz,
                   mh_kwargs=mapper_kwargs,
                   ch_kwargs=coach_kwargs,
                   preprocess_kwargs=preprocess_kwargs,
                   device=device) for i in range(n_models)]
    for model in models:
        model.preprocess = preprocess


    # Train models
    train_losses = []
    test_loss = []
    for i in tqdm(range(n_models), disable=True):
        train_losses += [models[i].train_loop(train_loader, n_epochs, lr=1e-2, disable_tqdm=True)]
        test_loss += [models[i].test_loop(test_loader)]
    train_losses = np.array(train_losses).T

    # Pick the best model on the test set
    ind_best_model = np.argmin(test_loss)
    best_model = models[ind_best_model]
    #
    valid_loss, x_pred, z_pred, hz_pred = best_model.valid_loop(test_loader)


    maxcs.append(get_maxes(*comp_ccorr(z_test, z_pred))[1])
    plt.plot(n_iter, maxcs[-1], 'o', color='blue')
    plt.draw()
    plt.pause(0.05)

df = save_results(fname='./maco_res.csv', r=maxcs, N=N, method='MaCo', dataset='lorenz')

plt.ioff()
plt.figure()
mngr = plt.get_current_fig_manager()
mngr.window.wm_geometry("+%d+%d" % (1000, 0))
plt.hist(maxcs)
plt.xlim(0, 1)
plt.show()