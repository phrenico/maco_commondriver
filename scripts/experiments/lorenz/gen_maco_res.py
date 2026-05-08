'''Apply MACO to the Lorenz system and plot the results.

'''
import torch
import torchvision.transforms as transforms

from functools import partial
import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from scripts.experiments.lorenz.config_lorenzres import interim_res_path, N, train_split, data_path_template, valid_split
from cdriver.savers.saver import save_results
from cdriver.evaluate.evalz import comp_ccorr, get_maxes
from cdriver.network.maco import MaCo
from scripts.experiments.maco_utils import build_loaders, get_default_device, train_and_select_best_model
from tqdm import tqdm
os.makedirs(interim_res_path, exist_ok=True)

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


plt.ion()
plt.figure(figsize=(10, 10))
plt.show()
mngr = plt.get_current_fig_manager()
mngr.window.wm_geometry("+%d+%d" % (0, 0))
plt.xlim(-1, 100)
plt.ylim(0, 1)


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
    plt.plot(n_iter, maxcs[-1], 'o', color='blue')
    plt.draw()
    plt.pause(0.05)

df = save_results(fname=interim_res_path / './maco_res.csv',
                  r=maxcs,
                  N=N,
                  method='MaCo',
                  dataset='lorenz')

# plt.ioff()
plt.figure()
mngr = plt.get_current_fig_manager()
mngr.window.wm_geometry("+%d+%d" % (1000, 0))
plt.hist(maxcs)
plt.xlim(0, 1)
plt.show()

plt.pause(0.05)
plt.close()