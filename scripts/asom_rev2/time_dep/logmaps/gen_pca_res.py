'''Script to run PCA on logistic map data-set
1. Generate data
2. Run PCA
3. Plot results
4. Save results

'''
import numpy as np
import sys
sys.path.append('./')
sys.path.append('../../../')
from sklearn.decomposition import PCA
from cdriver.preprocessing.splitters import train_valid_test_split
from cdriver.preprocessing.tde import time_delay_embedding
from cdriver.savers.saver import save_results
from cdriver.evaluate.evalz import comp_ccorr, get_maxes
from cdriver.datagen.logmap import gen_logmapdata

from scripts.datagen_scripts.datagen_config import logmapgen_params
from config_logmapres import train_split, interim_res_path, valid_split
import matplotlib.pyplot as plt

from tqdm import tqdm

if __name__ == "__main__":

    # 1. Generate data
    N = logmapgen_params['N']  # number of realizations
    dataset, params = gen_logmapdata(logmapgen_params)

    # 2. Run PCA
    d_embed = 3
    maxcs = []
    for i in tqdm(range(N)):
        X = time_delay_embedding(dataset[i][:, 1], delay=1, dimension=d_embed)
        Y = time_delay_embedding(dataset[i][:, 2], delay=1, dimension=d_embed)
        z = dataset[i][d_embed - 1:, 0]

        X_train, Y_train, z_train, X_valid, Y_valid, z_valid, X_test, Y_test, z_test = train_valid_test_split(X, Y, z, train_split, valid_split)

        D = np.concatenate([X_train, Y_train], axis=1)

        model = PCA(n_components=1).fit(D)
        zpred = model.transform(np.concatenate([X_test, Y_test], axis=1))

        maxcs.append(get_maxes(*comp_ccorr(z_test, zpred[:, 0]))[1])

    # Save results
    df = save_results(fname=interim_res_path / 'pca_res.csv',
                      r=maxcs,
                      N=N,
                      method='PCA',
                      dataset='logmap')
    

    # # 3. Plot results
    # plt.figure()
    # plt.hist(maxcs)
    # plt.show()
