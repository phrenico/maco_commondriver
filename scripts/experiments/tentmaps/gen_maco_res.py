'''Apply MaCo to the random Logistic datasets with the new version with preprocessing
0. Import packages
1. Generate random Logistic datasets
2. Apply MaCo to the datasets
3. Save the results
4. Plot the results
'''
import numpy as np
from tqdm import tqdm

from cdriver.network.maco import MaCo
from cdriver.savers.saver import save_results
from cdriver.datagen.tent_map import gen_tentmapdata

from scripts.datagen_scripts.datagen_config import tentmapgen_params
from scripts.experiments.tentmaps.config_tentmapres import train_split, interim_res_path, valid_split
import torch
import matplotlib.pyplot as plt
from scripts.experiments.maco_utils import (build_series_loaders,
                                       get_default_device,
                                       score_latent_reconstruction,
                                       train_and_select_best_model)

import matplotlib
matplotlib.use('TkAgg')

plt.ion()
plt.figure(figsize=(10, 10))
mngr = plt.get_current_fig_manager()
mngr.window.wm_geometry("+%d+%d" % (0, 0))
plt.show()

plt.xlim(-1, tentmapgen_params['N'] + 1)
plt.ylim(0, 1)


# 1. Generate random Logistic datasets
N = tentmapgen_params["N"]
dataset, params = gen_tentmapdata(tentmapgen_params)


# 2. Apply MaCo to the datasets
# define MaCo model
n_epochs = 300
n_models = 10
dx = 1
dy = 2
dz = 1
nh = 20  # number of hidden units
mapper_kwargs = dict(n_h1=nh, n_h2=nh)
coach_kwargs = dict(n_h1=nh, n_out=1)
preprocess_kwargs = dict(tau=1)
device = get_default_device()

maxcs = []
for n_iter in tqdm(range(N)):
    data = dataset[n_iter].astype(float)

    train_loader, test_loader, valid_loader, z_test = build_series_loaders(data,
                                                                           batch_size=1000,
                                                                           trainset_size=train_split * 100,
                                                                           testset_size=100 - (train_split + valid_split) * 100,
                                                                           validset_size=valid_split * 100)

    model_factory = lambda: MaCo(Ex=dx, Ey=dy, Ez=dz,
                                 mh_kwargs=mapper_kwargs,
                                 ch_kwargs=coach_kwargs,
                                 preprocess_kwargs=preprocess_kwargs,
                                 device=device)
    models, train_losses, valid_loss, best_model = train_and_select_best_model(model_factory,
                                                                               train_loader,
                                                                               valid_loader,
                                                                               n_models,
                                                                               n_epochs,
                                                                               1e-2)

    valid_loss, x_pred, z_pred, hz_pred = best_model.valid_loop(test_loader)
    maxcs.append(score_latent_reconstruction(z_pred, z_test))

    plt.plot(n_iter, maxcs[-1], 'o', color='blue')
    plt.draw()
    plt.pause(0.05)

# Save results
df = save_results(fname=interim_res_path / './maco_res.csv',
                  r=maxcs,
                  N=N,
                  method='MaCo',
                  dataset='tentmap')

# 3. Plot results
# plt.ioff()
plt.figure()
plt.hist(maxcs)
plt.show()
# print(maxcs)
plt.pause(1)
plt.close()