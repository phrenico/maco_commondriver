'''Run experiments with SFA

'''
import numpy as np
import matplotlib.pyplot as plt

import sksfa

from tqdm import tqdm
from sklearn.preprocessing import scale, PolynomialFeatures

import sys

sys.path.append('../')
# sys.path.append('../../../')
# sys.path.append('/home/phrenico/Projects/Codes/maco_commondriver/scripts_and_results/comparisons')
# sys.path.append('/home/phrenico/Projects/Codes/maco_commondriver')
sys.path.append('../../../')

from cdriver.preprocessing.splitters import train_valid_test_split
from cdriver.preprocessing.tde import time_delay_embedding
from cdriver.savers.saver import save_results
from cdriver.evaluate.evalz import comp_ccorr, get_maxes
from cdriver.datagen.tent_map import gen_tentmapdata

from scripts.datagen_scripts.datagen_config import tentmapgen_params
from config_tentmapres import train_split, interim_res_path, valid_split

import matplotlib
matplotlib.use('TkAgg')

if __name__ == "__main__":
    plt.ion()
    plt.figure(figsize=(10, 10))
    plt.show()
    mngr = plt.get_current_fig_manager()
    mngr.window.wm_geometry("+%d+%d" % (0, 0))
    plt.xlim(-1, tentmapgen_params['N'] + 1)
    plt.ylim(0, 1)

    N = tentmapgen_params['N']  # number of realizations
    dataset, params = gen_tentmapdata(tentmapgen_params)

    d_embed = 3
    maxcs = []
    for n_iter in tqdm(range(N)):
        data = dataset[n_iter]

        z = data[:-(d_embed - 1), 0]
        X = time_delay_embedding(data[:, 1], dimension=d_embed)
        Y = time_delay_embedding(data[:, 2], dimension=d_embed)

        X_train, Y_train, z_train, X_valid, Y_valid, z_valid, X_test, Y_test, z_test = train_valid_test_split(X, Y, z, 
                                                                                                              train_split,
                                                                                                              valid_split)
        D_train = np.concatenate([X_train, Y_train], axis=1)
        D_test = np.concatenate([X_test, Y_test], axis=1)

        # creating polynomial features
        poly = PolynomialFeatures(degree=2)
        D_train = poly.fit_transform(D_train)
        D_test = poly.transform(D_test)

        # 2. Run SFA
        sfa = sksfa.SFA(n_components=1)
        sfa.fit(D_train)
        zpred = sfa.transform(D_test).squeeze()

        tau, c = comp_ccorr(zpred, z_test)
        maxcs.append(get_maxes(tau, c)[1])

        plt.plot(n_iter, maxcs[-1], 'o', color='blue')
        plt.draw()
        plt.pause(0.05)

    # Save results
    df = save_results(fname=interim_res_path / 'sfa_res.csv',
                      r=maxcs,
                      N=N,
                      method='SFA',
                      dataset='tentmap')

    # # 3. Plot results
    plt.figure()
    plt.hist(maxcs)
    plt.show()
    plt.close()
