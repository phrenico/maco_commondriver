import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import sys
sys.path.append('../')
from config_lorenzres import interim_res_path, N, train_split, data_path_template, valid_split
from cdriver.preprocessing.splitters import train_valid_test_split
from cdriver.savers.saver import save_results
from cdriver.evaluate.evalz import comp_ccorr, get_maxes
from tqdm import tqdm
os.makedirs(interim_res_path, exist_ok=True)

import dca
DCA = dca.DynamicalComponentsAnalysis
matplotlib.use('TkAgg')

print(DCA.__name__)

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

    X = data['v'][:, 3:6]
    Y = data['v'][:, 6:]
    z = data['v'][:, 1]


    X_train, Y_train, z_train, X_valid, Y_valid, z_valid, X_test, Y_test, z_test = train_valid_test_split(X, Y, z,
                                                                                                          train_split,
                                                                                                          valid_split)

    D_train = np.concatenate([X_train, Y_train], axis=1)
    D_test = np.concatenate([X_test, Y_test], axis=1)

    # 2. Run DCA
    n_components = 5
    dca_model = DCA(d=n_components, T=5, n_init=10)
    dca_model.fit(D_train)
    z_pred = dca_model.transform(D_test)

    m = max([get_maxes(*comp_ccorr(z_test, z_pred[:, j]))[1] for j in range(n_components)])
    maxcs.append(m)

    plt.plot(n_iter, maxcs[-1], 'o', color='blue')
    plt.draw()
    plt.pause(0.05)

df = save_results(fname=interim_res_path / './dca_res.csv',
                  r=maxcs,
                  N=N,
                  method='DCA',
                  dataset='lorenz')

# plt.ioff()
plt.figure()
mngr = plt.get_current_fig_manager()
mngr.window.wm_geometry("+%d+%d" % (1000, 0))
plt.hist(maxcs)

plt.pause(2)
plt.close()