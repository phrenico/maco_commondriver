from unittest import TestCase

from matplotlib import pyplot as plt

from cdriver.network.anisom import AniSOM
import torch
import numpy as np

from scripts_and_results.comparisons.data_generators import LogmapExpRunner, comp_ccorr, get_maxes
from cdriver.preprocessing.tde import TimeDelayEmbeddingTransform
from sklearn.preprocessing import scale
from mpl_toolkits.mplot3d import Axes3D

class TestAniSOM(TestCase):
    def test_call(self):
        d_grid = 2
        d_space = 3
        sizes = [20, 10]
        ani = AniSOM(space_dim=d_space, grid_dim=d_grid, sizes=sizes)



        # Generate data
        n = 10000
        rint = (3.8, 4.)  # interval to chose from the value of r parameter
        A0 = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]])  # basic connection structure
        A = np.array([[1., 0., 0.],
                      [0.3, 1., 0.],
                      [0.4, 0., 1.]])
        i = 123
        data = LogmapExpRunner(nvars=3, baseA=A0, r_interval=rint).gen_experiment(n=n, A=A, seed=i)[0]

        x = torch.Tensor(data[:, 1:2])
        y = torch.Tensor(data[:, 2:3])

        tde = TimeDelayEmbeddingTransform(embedding_dim=d_space, delay=1)
        X = tde(x)
        Y = tde(y)
        z = data[:-(d_space - 1), 0]


        # print(ani(x).shape, ani(x[0]).shape)


        ani.fit(X, Y, epochs=4)
        x_pred = ani.predict(X)

        # print("shape of X:", X.shape, "shape of predicted coordinates:", x_pred.shape)

        tau, c = comp_ccorr(x_pred[:, 0], z)
        print(get_maxes(tau, c)[1])
        tau, c = comp_ccorr(x_pred[:, 1], z)
        print(get_maxes(tau, c)[1])



        # plt.plot(ani.epss)
        # print(ani.epss)


        plt.figure()
        plt.subplot(111, projection='3d')
        plt.plot(*X.T, '.', alpha=0.2)

        print(ani.grid[0, :, :].shape)
        [plt.plot( *ani.grid[i, :, :].T, 'b-') for i in range(ani.grid.shape[0]) ]
        [plt.plot(*ani.grid[:, i, :].T, 'o-') for i in range(ani.grid.shape[1])]



        plt.show()