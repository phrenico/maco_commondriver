'''Run Shrec experiments'''
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
sys.path.append('./')
sys.path.append('../../../')

from shrec.models import RecurrenceManifold
from cdriver.preprocessing.splitters import train_valid_test_split
from cdriver.preprocessing.tde import time_delay_embedding
from cdriver.savers.saver import save_results
from cdriver.evaluate.evalz import comp_ccorr, get_maxes
from cdriver.datagen.logmap import gen_logmapdata

from scripts.datagen_scripts.datagen_config import logmapgen_params
from config_logmapres import train_split, interim_res_path, valid_split

# @title Fixed Coupling
N = logmapgen_params['N']  # number of realizations
dataset, params = gen_logmapdata(logmapgen_params)


#Run the  Reconstructions on the Datasets
maxcs = []
for i in tqdm(range(N)):
  data = dataset[i]
  X = data[:, 1:]
  y = data[:, 0]

  X_train, _, z_train, X_valid, _, z_valid, X_test, __, z_test = train_valid_test_split(X, X, y, train_split, valid_split)


  model = RecurrenceManifold(d_embed=3)

  y_recon = model.fit_predict(X_train)
  # model.fit(X_train)
  # y_recon = model.transform(X_test)

  tau, c = comp_ccorr(z_train, y_recon)
  maxtau, maxc = get_maxes(tau, c)
  maxcs.append(maxc)

# Save results
df = save_results(fname=interim_res_path / 'shrec_res.csv',
                  r=maxcs,
                  N=N,
                  method='ShRec',
                  dataset='logmap')

# # 3. Plot results
# plt.hist(maxcs)
# plt.show()