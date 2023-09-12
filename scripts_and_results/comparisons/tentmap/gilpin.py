'''Run Shrec experiments'''
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
sys.path.append('../')
sys.path.append('../../../')

from data_generators import  comp_ccorr, get_maxes, train_valid_test_split, save_results, time_delay_embedding
from cdriver.datagen.tent_map import TentMapExpRunner
from shrec.models import RecurrenceManifold


# @title Fixed Coupling
from data_config import N, n, A0, aint

import matplotlib
matplotlib.use('TkAgg')

plt.ion()
plt.figure(figsize=(10, 10))
mngr = plt.get_current_fig_manager()
mngr.window.wm_geometry("+%d+%d" % (0, 0))
plt.show()
plt.xlim(-1, 100)
plt.ylim(0, 1)

dataset = [TentMapExpRunner(nvars=3, baseA=A0, a_interval=aint).gen_experiment(n=n, seed=i)[0] for i in tqdm(range(N))]



#Run the  Reconstructions on the Datasets
train_split = 0.8
valid_split = 0.1
maxcs = []
for n_iter in tqdm(range(N)):
  data = dataset[n_iter]
  X = data[:, 1:]
  y = data[:, 0]

  X_train, _, z_train, X_valid, _valid, z_valid, X_test, __, z_test = train_valid_test_split(X, X, y, train_split, valid_split)


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
df = save_results(fname='./shrec_res.csv', r=maxcs, N=N, method='ShRec', dataset='tentmap')

# 3. Plot results
plt.ioff()
plt.figure()
plt.hist(maxcs)
plt.show()