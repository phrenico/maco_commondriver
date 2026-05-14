import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import scale
from tqdm.auto import tqdm

from cdriver.preprocessing.splitters import train_valid_test_split
from cdriver.preprocessing.tde import time_delay_embedding
from cdriver.savers.saver import save_results
from cdriver.evaluate.evalz import comp_ccorr, get_maxes
from cdriver.datagen.tent_map import gen_tentmapdata

from scripts.datagen_scripts.datagen_config import tentmapgen_params
from scripts.experiments.tentmaps.config_tentmapres import train_split, interim_res_path, valid_split

N = tentmapgen_params['N']  # number of realizations
dataset, params = gen_tentmapdata(tentmapgen_params)

print('Generated {} realizations of logistic map data-set with parameters: {}'.format(N, params))
print("Training-Data length will be: {}".format(int(tentmapgen_params['n'] * train_split)))


d_embed = 2
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

    n_components = 1
    pca =  KernelPCA(n_components=n_components, kernel='rbf').fit(D_train)
    zpred = pca.transform(D_test)

    m = max([get_maxes(*comp_ccorr(z_test, zpred[:, j]))[1] for j in range(n_components)])
    maxcs.append(m)

df = save_results(fname=interim_res_path / 'kpca_res.csv',
                  r=maxcs,
                  N=N,
                  method='KPCA',
                  dataset='tentmap',
                  times= N * ['NaN'])


