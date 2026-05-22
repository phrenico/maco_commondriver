"""Generates the example results on the logistic map example
"""
import argparse
import os

import numpy as np
from tqdm import tqdm

from cdriver.network.maco import MaCo
from cdriver.evaluate.evalz import comp_ccorr, get_maxes
from cdriver.datagen.logmap import gen_logmapdata

import torch
import pandas as pd

from pathlib import Path
import pickle

from scripts.experiments.config_loader import get_config, resolve_paths
from scripts.experiments.maco_utils import build_series_loaders, train_and_select_best_model

_REPO_ROOT = Path(__file__).resolve().parents[3]


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='scripts/config_runall.py')
    args = parser.parse_args(args)
    cfg = resolve_paths(get_config('example_logmap', args.config), _REPO_ROOT)

    respath = cfg['paths']['final_res_path']
    os.makedirs(respath, exist_ok=True)

    # Parameters
    dx = cfg['maco']['dx']
    dy = cfg['maco']['dy']
    dz = cfg['maco']['dz']
    nh = cfg['maco']['n_hidden']
    mapper_kwargs = dict(n_h1=nh, n_h2=nh)
    coach_kwargs = dict(n_h1=nh, n_out=1)
    preprocess_kwargs = dict(tau=cfg['maco']['tau'])
    device = cfg['maco']['device']

    n_models = cfg['maco']['n_models']

    trainset_size = cfg['preprocessing']['trainset_size']
    testset_size = cfg['preprocessing']['testset_size']
    validset_size = cfg['preprocessing']['validset_size']

    n_epochs = cfg['maco']['n_epochs']
    batch_size = cfg['maco']['batch_size']
    lr = cfg['maco']['lr']

    dataset, params = gen_logmapdata(cfg['datagen'])
    data = dataset[0]
    np.savez(respath / 'data_params.npz', params=params[0], dataset=data)

    train_loader, test_loader, valid_loader, z_test = build_series_loaders(data,
                                                                           batch_size=batch_size,
                                                                           trainset_size=trainset_size,
                                                                           testset_size=testset_size,
                                                                           validset_size=validset_size)
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
                                                                               lr,
                                                                               disable_tqdm=False)

    # valid_loss, x_pred, z_pred, hz_pred = best_model.valid_loop(valid_loader)
    test_loss, x_pred, z_pred, hz_pred = best_model.valid_loop(test_loader)

    tau, c = comp_ccorr(z_pred, z_test)

    maxcs = [get_maxes(tau, c)[1], ]

    # compute the correlation between the hidden variables
    r_reconst = []
    r_predict = []
    for model in tqdm(models):
        preds = model.valid_loop(test_loader)
        r_predict += [np.corrcoef(preds[1], test_loader[0].squeeze()[1:])[0, 1]]
        r_reconst += [np.corrcoef(preds[2], z_test.squeeze()[:-1])[0, 1]]

    # Save out results
    res_dict = {'cc_pred': z_pred,
                'cc_test': z_test[:-1],
                'x_test': test_loader[0].squeeze().detach().numpy()[1:],
                'x_past_test': test_loader[0].squeeze()[:-1],
                'x_pred': x_pred,
                'Y_1_test': test_loader[1].squeeze()[:-1],
                'Y_2_test': test_loader[1].squeeze()[1:],
                }

    df = pd.DataFrame(res_dict)

    df.to_csv(respath / 'mappercoach_res.csv')
    np.save(respath / 'learning_curves.npy', train_losses)
    np.save(respath / 'valid_loss.npy', valid_loss)
    torch.save(best_model, respath / 'best_model.pth')
    with open(respath / 'models.pkl', 'wb') as f:
        pickle.dump(models, f)
    pd.DataFrame({'r_predict': r_predict,
                  'r_reconst': r_reconst}).to_csv(respath / 'r_values.csv')


if __name__ == '__main__':
    main()
