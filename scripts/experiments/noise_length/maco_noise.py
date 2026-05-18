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

from cdriver.datagen.logmap import LogmapExpRunner
from cdriver.network.maco import MaCo

import torch

from scripts.experiments.config_loader import get_config, resolve_paths
from scripts.experiments.maco_utils import (build_series_loaders,
                                       create_sweep_df,
                                       get_default_device,
                                       score_latent_reconstruction,
                                       train_and_select_best_model)
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None)
    _args = parser.parse_args()
    cfg = resolve_paths(get_config('noise_length', _args.config), _REPO_ROOT)
else:
    cfg = resolve_paths(get_config('noise_length', None), _REPO_ROOT)

_maco = dict(cfg['maco'])
_maco['nh'] = _maco.pop('n_hidden')
noise_params = {**cfg['datagen'], **cfg['noise_sweep'], **_maco, **cfg['preprocessing']}
p = SimpleNamespace(**noise_params)
final_res_path = cfg['paths']['final_res_path']
os.makedirs(final_res_path, exist_ok=True)

# 1. Generate random Logistic datasets
datasets, params = zip(
    *[LogmapExpRunner(nvars=p.nvars,
                      baseA=p.A0,
                      r_interval=p.rint).gen_experiment(n=p.n,
                                                        seed=i) for i in tqdm(range(p.N))])


print("Data sigma: ",np.mean([i.std() for i in datasets]))

mapper_kwargs = dict(n_h1=p.nh, n_h2=p.nh)
coach_kwargs = dict(n_h1=p.nh, n_out=1)
preprocess_kwargs = dict(tau=p.tau)
device = get_default_device()

maxdict = {}
for L in tqdm(p.Ls, desc='Noise levels'):
    maxcs = []
    for n_iter in tqdm(range(p.N), desc='instances', leave=False):
        data = datasets[n_iter].astype(float)
        data[:, 1:] = data[:, 1:] + np.random.normal(loc=0, scale=L, size=data[:, 1:].shape)  # add observation noise to the observed time series

        train_loader, test_loader, valid_loader, z_test = build_series_loaders(data,
                                                                               batch_size=p.batch_size,
                                                                               trainset_size=p.trainset_size,
                                                                               testset_size=p.testset_size,
                                                                               validset_size=p.validset_size)
        model_factory = lambda: MaCo(Ex=p.dx, Ey=p.dy, Ez=p.dz,
                                     mh_kwargs=mapper_kwargs,
                                     ch_kwargs=coach_kwargs,
                                     preprocess_kwargs=preprocess_kwargs,
                                     device=device)
        models, train_losses, valid_loss, best_model = train_and_select_best_model(model_factory,
                                                                                   train_loader,
                                                                                   valid_loader,
                                                                                   p.n_models,
                                                                                   p.n_epochs,
                                                                                   p.lr,
                                                                                   desc='Models',
                                                                                   leave=False)

        valid_loss, x_pred, z_pred, hz_pred = best_model.valid_loop(test_loader)
        maxcs.append(score_latent_reconstruction(z_pred, z_test))
    maxdict[L] = maxcs.copy()

df = create_sweep_df(maxdict)
df.to_csv(final_res_path / './noise_maco_res.csv')