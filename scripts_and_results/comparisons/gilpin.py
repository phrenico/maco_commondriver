'''Run Shrec experiments'''
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from data_generators import LogmapExpRunner, comp_ccorr, get_maxes, train_test_split, save_results, time_delay_embedding
from shrec.models import RecurrenceManifold


# @title Fixed Coupling
n = 20_000
N = 50
rint = (3.8, 4.)  # interval to chose from the value of r parameter
A0 = np.array([[0, 0, 0],[1, 0, 0], [1, 0, 0]])  # basic connection structure
A = np.array([ [1., 0., 0.],
             [0.3, 1., 0.],
             [0.4, 0., 1.]])
train_split = 0.5

dataset = [LogmapExpRunner(nvars=3, baseA=A0, r_interval=rint).gen_experiment(n=n, A=A, seed=i)[0] for i in tqdm(range(N))]
params = [LogmapExpRunner(nvars=3, baseA=A0, r_interval=rint).gen_experiment(n=2, A=A, seed=i)[1] for i in range(N)]


#Run the  Reconstructions on the Datasets

maxcs = []
for i in tqdm(range(N)):
  data = dataset[i]
  X = data[:, 1:]
  y = data[:, 0]

  X_train, _, z_train, X_test, __, z_test = train_test_split(X, X, y, train_split)


  model = RecurrenceManifold(d_embed=3)

  y_recon = model.fit_predict(X_train)
  # model.fit(X_train)
  # y_recon = model.transform(X_test)

  tau, c = comp_ccorr(z_train, y_recon)
  maxtau, maxc = get_maxes(tau, c)
  maxcs.append(maxc)

# Save results
df = save_results(fname='./shrec_res.csv', r=maxcs, N=N, method='ShRec', dataset='logmap_fixed')

# 3. Plot results
plt.hist(maxcs)
plt.show()