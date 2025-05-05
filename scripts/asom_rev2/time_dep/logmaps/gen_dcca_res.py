"""It's not running, use Google Colab in stead"""
import torch
# set default device to cpu
# cpu
torch.device('cpu')

import numpy as np

from tqdm import tqdm
import sys
sys.path.append('./')
sys.path.append('../../../')
from mvlearn.embed import DCCA
from cdriver.preprocessing.splitters import train_valid_test_split
from cdriver.preprocessing.tde import time_delay_embedding
from cdriver.savers.saver import save_results
from cdriver.evaluate.evalz import comp_ccorr, get_maxes
from cdriver.datagen.logmap import gen_logmapdata

from scripts.datagen_scripts.datagen_config import logmapgen_params
from config_logmapres import train_split, interim_res_path, valid_split


import sys

sys.path.append('../')

import pandas as pd


def myfun(x, *args, **kwargs):
    return torch.linalg.eigh(x)


torch.symeig = myfun  # redefine function to make it work with dcca

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# 1. Generate data
N = logmapgen_params['N']  # number of realizations
dataset, params = gen_logmapdata(logmapgen_params)

# Define parameters and layers for deep model
d_embed = 3
features1 = d_embed  # Feature sizes
features2 = d_embed
layers1 = [20, 20, 1]  # nodes in each hidden layer and the output size
layers2 = layers1.copy()

maxcs = []
maxcs2 = []
maxcs3 = []

for i in tqdm(range(N)):
    data = dataset[i]

    z = data[:-(d_embed - 1), 0]
    X = time_delay_embedding(data[:, 1], dimension=d_embed)
    Y = time_delay_embedding(data[:, 2], dimension=d_embed)

    X_train, Y_train, z_train, X_valid, Y_valid, z_valid, X_test, Y_test, z_test = train_valid_test_split(X, Y, z,
                                                                                                          train_split,
                                                                                                          valid_split)

    dcca = DCCA(input_size1=features1, input_size2=features2, n_components=1,
                layer_sizes1=layers1, layer_sizes2=layers2, epoch_num=500,
                use_all_singular_values=True, device=device)
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
df = save_results(fname=interim_res_path / 'dcca_res.csv',
                  r=maxcs3,
                  N=N,
                  method='DCCA',
                  dataset='logmap')
