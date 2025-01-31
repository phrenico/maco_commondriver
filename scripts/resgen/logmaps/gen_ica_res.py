'''Script to run ICA on logistic map data-set
1. Generate data
2. Run ICA
3. Plot results
4. Save results

'''
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('./')
sys.path.append('../../../')
from sklearn.decomposition import FastICA
from cdriver.preprocessing.splitters import train_valid_test_split
from cdriver.preprocessing.tde import time_delay_embedding
from cdriver.savers.saver import  save_results
from cdriver.evaluate.evalz import comp_ccorr, get_maxes
from cdriver.datagen.logmap import gen_logmapdata

from scripts.datagen_scripts.datagen_config import logmapgen_params
from config_logmapres import train_split, interim_res_path, valid_split

from tqdm import tqdm

if __name__=="__main__":

    #1. Generate data
    N = logmapgen_params['N']  # number of realizations
    dataset, params = gen_logmapdata(logmapgen_params)


    #2. Run ICA
    d_embed = 3
    maxcs = []
    for i in tqdm(range(N)):
        X = time_delay_embedding(dataset[i][:, 1], delay=1, dimension=d_embed)
        Y = time_delay_embedding(dataset[i][:, 2], delay=1, dimension=d_embed)
        z = dataset[i][d_embed-1:, 0]
        X_train, Y_train, z_train, X_valid, Y_valid, z_valid, X_test, Y_test, z_test = train_valid_test_split(X, Y, z, train_split, valid_split)

        D = np.concatenate([X_train, Y_train], axis=1)

        n_components = 2
        model = FastICA(n_components=n_components).fit(D)
        zpred = model.transform(np.concatenate([X_test, Y_test], axis=1))

        m = max([get_maxes(*comp_ccorr(z_test, zpred[:, j]))[1] for j in range(n_components)])

        maxcs.append(m)

    # Save results
    df = save_results(fname= interim_res_path / 'ica_res.csv',
                      r=maxcs,
                      N=N,
                      method='ICA',
                      dataset='logmap')

    # #3. Plot results
    # plt.figure()
    # plt.hist(maxcs)
    # plt.show()