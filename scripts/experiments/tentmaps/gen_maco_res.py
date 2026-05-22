'''Apply MaCo to the random tent map datasets'''
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from cdriver.network.maco import MaCo
from cdriver.savers.saver import save_results
from cdriver.datagen.tent_map import gen_tentmapdata
import torch

from scripts.experiments.config_loader import get_config, resolve_paths
from scripts.experiments.maco_utils import (build_series_loaders,
                                            get_default_device,
                                            score_latent_reconstruction,
                                            train_and_select_best_model)

_REPO_ROOT = Path(__file__).resolve().parents[3]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='scripts/config_runall.py')
    args = parser.parse_args()
    cfg = resolve_paths(get_config('tentmaps', args.config), _REPO_ROOT)

    N = cfg['datagen']['N']
    dataset, params = gen_tentmapdata(cfg['datagen'])

    train_split = cfg['preprocessing']['train_split']
    valid_split = cfg['preprocessing']['valid_split']
    test_split = 1.0 - train_split - valid_split

    n_epochs = cfg['maco']['n_epochs']
    n_models = cfg['maco']['n_models']
    bs = cfg['maco']['batch_size']
    lr = cfg['maco']['lr']
    dx = cfg['maco']['dx']
    dy = cfg['maco']['dy']
    dz = cfg['maco']['dz']
    nh = cfg['maco']['n_hidden']
    tau = cfg['maco']['tau']

    mapper_kwargs = dict(n_h1=nh, n_h2=nh)
    coach_kwargs = dict(n_h1=nh, n_out=1)
    preprocess_kwargs = dict(tau=tau)
    device = get_default_device()

    maxcs = []
    for n_iter in tqdm(range(N)):
        data = dataset[n_iter].astype(float)

        train_loader, test_loader, valid_loader, z_test = build_series_loaders(
            data,
            batch_size=bs,
            trainset_size=train_split * 100,
            testset_size=100 - (train_split + valid_split) * 100,
            validset_size=valid_split * 100,
        )

        model_factory = lambda: MaCo(Ex=dx, Ey=dy, Ez=dz,
                                     mh_kwargs=mapper_kwargs,
                                     ch_kwargs=coach_kwargs,
                                     preprocess_kwargs=preprocess_kwargs,
                                     device=device)
        models, train_losses, valid_loss, best_model = train_and_select_best_model(
            model_factory, train_loader, valid_loader, n_models, n_epochs, lr)

        valid_loss, x_pred, z_pred, hz_pred = best_model.valid_loop(test_loader)
        maxcs.append(score_latent_reconstruction(z_pred, z_test))

    df = save_results(fname=cfg['paths']['interim_res_path'] / 'maco_res.csv',
                      r=maxcs, N=N, method='MaCo', dataset='tentmap')
