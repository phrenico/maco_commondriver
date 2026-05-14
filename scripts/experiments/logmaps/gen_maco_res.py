'''Apply MaCo to the random Logistic datasets with the new version with preprocessing (more epochs [300], smaller batch size [1_000])
0. Import packages
1. Generate random Logistic datasets
2. Apply MaCo to the datasets
3. Save the results
4. Plot the results
'''
import numpy as np
from tqdm import tqdm

from cdriver.network.maco import MaCo
from cdriver.savers.saver import  save_results
from cdriver.datagen.logmap import gen_logmapdata

import torch


from scripts.datagen_scripts.datagen_config import logmapgen_params
from scripts.experiments.logmaps.config_logmapres import interim_res_path, train_split, valid_split, test_split
from scripts.experiments.maco_utils import (build_series_loaders,
                                       get_default_device,
                                       score_latent_reconstruction,
                                       train_and_select_best_model)



# 1. Generate random Logistic datasets
N = logmapgen_params["N"]
dataset, params = gen_logmapdata(logmapgen_params)


# 2. Apply MaCo to the datasets
# define MaCo model
n_epochs = 300
n_models = 10
bs = 1_000
lr = 1e-2
dx = 1
dy = 2
dz = 1
nh = 20  # number of hidden units
mapper_kwargs = dict(n_h1=nh, n_h2=nh)
coach_kwargs = dict(n_h1=nh, n_out=1)
preprocess_kwargs = dict(tau=1)
device = get_default_device()
print("device: ", device)

maxcs = []
for n_iter in tqdm(range(N)):
    data = dataset[n_iter].astype(float)

    train_loader, test_loader, valid_loader, z_test = build_series_loaders(data,
                                                                           batch_size=bs,
                                                                           trainset_size=int(train_split * 100),
                                                                           testset_size=int(100 * test_split),
                                                                           validset_size=int(valid_split * 100))

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
                                                                               lr)

    valid_loss, x_pred, z_pred, hz_pred = best_model.valid_loop(test_loader)
    maxcs.append(score_latent_reconstruction(z_pred, z_test))

# Save results
df = save_results(fname=interim_res_path / 'maco_res.csv',
                  r=maxcs,
                  N=N,
                  method='MaCo',
                  dataset='logmap')
