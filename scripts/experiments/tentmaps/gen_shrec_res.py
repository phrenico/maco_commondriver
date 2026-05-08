'''Run Shrec experiments'''

import matplotlib.pyplot as plt
from tqdm import tqdm

from cdriver.preprocessing.splitters import train_valid_test_split

from cdriver.savers.saver import save_results
from cdriver.evaluate.evalz import comp_ccorr, get_maxes
from cdriver.datagen.tent_map import gen_tentmapdata

from scripts.datagen_scripts.datagen_config import tentmapgen_params
from scripts.experiments.tentmaps.config_tentmapres import train_split, interim_res_path, valid_split
from shrec.models import RecurrenceManifold

# @title Fixed Coupling


import matplotlib

matplotlib.use('TkAgg')

plt.ion()
plt.figure(figsize=(10, 10))
mngr = plt.get_current_fig_manager()
mngr.window.wm_geometry("+%d+%d" % (0, 0))
plt.show()
plt.xlim(-1, tentmapgen_params['N'] + 1)
plt.ylim(0, 1)

N = tentmapgen_params['N']  # number of realizations
dataset, params = gen_tentmapdata(tentmapgen_params)

# Run the  Reconstructions on the Datasets

maxcs = []
for n_iter in tqdm(range(N)):
    data = dataset[n_iter]
    X = data[:, 1:]
    y = data[:, 0]

    X_train, _, z_train, X_valid, _valid, z_valid, X_test, __, z_test = train_valid_test_split(X, X, y, train_split,
                                                                                               valid_split)

    model = RecurrenceManifold(d_embed=3)

    y_recon = model.fit_predict(X_train)
    # model.fit(X_train)
    # y_recon = model.predict(X_test)

    tau, c = comp_ccorr(z_train, y_recon)
    maxtau, maxc = get_maxes(tau, c)
    maxcs.append(maxc)

    plt.plot(n_iter, maxcs[-1], 'o', color='blue')
    plt.draw()
    plt.pause(0.05)

# Save results
df = save_results(fname=interim_res_path / './shrec_res.csv',
                  r=maxcs,
                  N=N,
                  method='ShRec',
                  dataset='tentmap')

# 3. Plot results
# plt.ioff()
plt.figure()
plt.hist(maxcs)
plt.show()
plt.pause(1)
plt.close()
