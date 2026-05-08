import unittest

import numpy as np
import torch

from cdriver.datagen.logmap import LogmapExpRunner
from cdriver.network.anisom import AniSOM
from cdriver.preprocessing.tde import TimeDelayEmbeddingTransform


class TestAniSOM(unittest.TestCase):
    def test_fit_and_predict_shapes(self):
        np.random.seed(0)
        torch.manual_seed(0)

        base_a = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]])
        coupling = np.array([[1.0, 0.0, 0.0], [0.3, 1.0, 0.0], [0.4, 0.0, 1.0]])
        runner = LogmapExpRunner(nvars=3, baseA=base_a, r_interval=(3.8, 4.0))
        data, _ = runner.gen_experiment(n=32, A=coupling, seed=123)

        x = torch.tensor(data[:, 1:2], dtype=torch.float32)
        y = torch.tensor(data[:, 2:3], dtype=torch.float32)
        embedding = TimeDelayEmbeddingTransform(embedding_dim=3, delay=1)
        x_embedded = embedding(x)
        y_embedded = embedding(y)

        ani = AniSOM(space_dim=3, grid_dim=2, sizes=[6, 4])
        ani.K = 5
        ani.fit(x_embedded, y_embedded, epochs=1, disable_tqdm=True)

        activations = ani.forward(x_embedded[:2], squeeze=False)
        coordinates = ani.predict(x_embedded[:10])

        self.assertEqual(tuple(ani.grid.shape), (6, 4, 3))
        self.assertEqual(tuple(activations.shape), (2, 6, 4))
        self.assertEqual(tuple(coordinates.shape), (10, 2))
        self.assertEqual(len(ani.epss), x_embedded.shape[0])
        self.assertFalse(torch.isnan(ani.grid).any().item())
        self.assertTrue(torch.all(coordinates[:, 0] >= 0).item())
        self.assertTrue(torch.all(coordinates[:, 0] < ani.sizes[0]).item())
        self.assertTrue(torch.all(coordinates[:, 1] >= 0).item())
        self.assertTrue(torch.all(coordinates[:, 1] < ani.sizes[1]).item())