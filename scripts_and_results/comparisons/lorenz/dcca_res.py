import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import scale
import sys
sys.path.append('../')

from mvlearn.embed import DCCA

from torch.utils.data import DataLoader, TensorDataset
from data_generators import comp_ccorr, get_maxes, save_results, train_test_split
import torch
from tqdm import tqdm

def myfun(x, *args, **kwargs):
  return torch.linalg.eigh(x)

torch.symeig = myfun


plt.ion()
plt.figure(figsize=(10, 10))
mngr = plt.get_current_fig_manager()
mngr.window.wm_geometry("+%d+%d" % (0, 0))
plt.show()
plt.xlim(-1, 100)
plt.ylim(0, 1)

N = 100
train_split = 0.5
maxcs = []
for n_iter in tqdm(range(N)):
    data_path = '../../../data/lorenz/lorenz_{}.npz'.format(n_iter)
    data = np.load(data_path)

    X = data['v'][:, 3:6]
    Y = data['v'][:, 6:]
    z = data['v'][:, 1]


    d_embed = 3
    features1 = d_embed  # Feature sizes
    features2 = d_embed
    layers1 = [20, 20, 1]  # nodes in each hidden layer and the output size
    layers2 = layers1.copy()


    train_split = 0.5
    X_train, Y_train, z_train, X_test, Y_test, z_test = train_test_split(X, Y, z, train_split)

    dcca = DCCA(input_size1=features1, input_size2=features2, n_components=1,
                        layer_sizes1=layers1, layer_sizes2=layers2, epoch_num=100,
                        use_all_singular_values=True)
    dcca.fit([X_train, Y_train])
    Xs_transformed = dcca.transform([X_test, Y_test])

    zp1, zp2 = Xs_transformed

    z_pred = (zp1[:, 0] + zp2[:, 0]) / 2
    maxcs.append(get_maxes(*comp_ccorr(z_test, z_pred))[1])

    plt.plot(n_iter, maxcs[-1], 'o', color='blue')
    plt.draw()
    plt.pause(0.05)

df = save_results(fname='./dcca_res.csv', r=maxcs, N=N, method='DCCA', dataset='lorenz')

plt.ioff()
plt.figure()
mngr = plt.get_current_fig_manager()
mngr.window.wm_geometry("+%d+%d" % (1000, 0))
plt.hist(maxcs)
plt.xlim(0, 1)
plt.show()