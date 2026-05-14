"""Generates the example results on the logistic map example
"""
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

from scripts.config import example_logmap_final_res_path
from scripts.datagen_scripts.datagen_config import logmapexamplegen_params
from scripts.experiments.maco_utils import build_series_loaders, train_and_select_best_model


def main():
    respath = example_logmap_final_res_path
    os.makedirs(respath, exist_ok=True)

    # Parameters
    dx = 1
    dy = 2
    dz = 1
    nh = 20 # number of hidden units
    mapper_kwargs = dict(n_h1=nh, n_h2=nh)
    coach_kwargs = dict(n_h1=nh, n_out=1)
    preprocess_kwargs = dict(tau=1)
    device= "cpu"  # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    n_models = 10  # number of models to train

    trainset_size = 80
    testset_size = 10
    validset_size = 10

    n_epochs = 2_000
    batch_size = 1_000
    lr = 1e-2

    dataset, params = gen_logmapdata(logmapexamplegen_params)
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
    

    # print("shape of z_pred: {} and the shape of z_test: {}".format(z_pred.shape, z_test.shape))
    tau, c = comp_ccorr(z_pred, z_test)

    maxcs = [get_maxes(tau, c)[1], ]


    # compute the correlation between the hidden variables
    r_reconst = []
    r_predict = []
    for model in tqdm(models):
        preds = model.valid_loop(test_loader)
        # print(preds[1].shape, preds[2].shape, test_loader[1].squeeze().shape)
        r_predict += [np.corrcoef(preds[1], test_loader[0].squeeze()[1:])[0, 1] ]
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
    # for label, value in res_dict.items():
    #     print(label, value.shape)

    df = pd.DataFrame(res_dict)

    df.to_csv(respath / 'mappercoach_res.csv')
    np.save(respath / 'learning_curves.npy', train_losses)
    np.save(respath / 'valid_loss.npy', valid_loss)
    torch.save(best_model, respath / 'best_model.pth')
    with open(respath / 'models.pkl', 'wb') as f:
        pickle.dump(models, f)
    pd.DataFrame({'r_predict':r_predict,
                  'r_reconst':r_reconst}).to_csv(respath / 'r_values.csv')
    # print(np.corrcoef(z_test[:-1], z_pred))

if __name__ == "__main__":
    main()
