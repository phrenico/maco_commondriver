from shrec.models import RecurrenceManifold

import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from scripts.experiments.lorenz.config_lorenzres import interim_res_path, N, train_split, data_path_template, valid_split
from cdriver.preprocessing.splitters import train_valid_test_split
from cdriver.savers.saver import save_results
from cdriver.evaluate.evalz import comp_ccorr, get_maxes
from tqdm import tqdm
os.makedirs(interim_res_path, exist_ok=True)


plt.ion()
plt.figure(figsize=(10, 10))
plt.show()
mngr = plt.get_current_fig_manager()
mngr.window.wm_geometry("+%d+%d" % (0, 0))
plt.xlim(-1, 100)
plt.ylim(0, 1)

maxcs = []
for n_iter in tqdm(range(N)):
    data_path = data_path_template.format(n_iter)
    data = np.load(data_path)

    X = data['v'][:, 3:]
    z = data['v'][:, 1]
    T = X.shape[0]

    X_train, Y_train, z_train, X_valid, Y_valid, z_valid, X_test, Y_test, z_test = train_valid_test_split(X, X, z,
                                                                                                          train_split,
                                                                                                          valid_split)

    d_embed = 3
    model = RecurrenceManifold(d_embed=d_embed)
    z_pred = model.fit_predict(X_test)

    maxcs.append(get_maxes(*comp_ccorr(z_test, z_pred))[1])
    plt.plot(n_iter, maxcs[-1], 'o', color='blue')
    plt.draw()
    plt.pause(0.05)

df = save_results(fname=interim_res_path / './shrec_res.csv',
                  r=maxcs,
                  N=N,
                  method='ShRec',
                  dataset='lorenz')


plt.figure()
mngr = plt.get_current_fig_manager()
mngr.window.wm_geometry("+%d+%d" % (1000, 0))
plt.hist(maxcs)
plt.xlim(0, 1)
plt.show()

plt.pause(2)
plt.close()
