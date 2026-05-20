"""Integration smoke test: run one realization of each family end-to-end.

Uses small N and minimal epochs to verify that the pipeline components
(data generation, model fitting, CSV output) work together.
"""
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cdriver.datagen.logmap import gen_logmapdata
from cdriver.datagen.tent_map import gen_tentmapdata
from cdriver.network.maco import MaCo
from cdriver.savers.saver import save_results
from cdriver.evaluate.evalz import comp_ccorr, get_maxes
from cdriver.preprocessing.splitters import train_valid_test_split
from cdriver.preprocessing.tde import time_delay_embedding
from scripts.experiments.maco_utils import build_series_loaders, train_and_select_best_model
from scripts.experiments.method_runner import run_baseline_method
from sklearn.decomposition import PCA


class TestSmokeLogmap(unittest.TestCase):
    """End-to-end smoke test: logmap PCA baseline."""

    def test_logmap_pca_single_realization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            interim = Path(tmpdir) / 'interim'
            interim.mkdir()

            param_dict = dict(
                N=1, n=200, rint=(3.8, 4.0),
                A0=np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=float),
                A=np.array([[1.0, 0.0, 0.0], [0.3, 1.0, 0.0], [0.4, 0.0, 1.0]], dtype=float),
            )
            datasets, _params = gen_logmapdata(param_dict)
            data = datasets[0]

            d_embed = 3
            X = time_delay_embedding(data[:, 1], delay=1, dimension=d_embed)
            Y = time_delay_embedding(data[:, 2], delay=1, dimension=d_embed)
            z = data[d_embed - 1:, 0]

            X_tr, Y_tr, z_tr, X_va, Y_va, z_va, X_te, Y_te, z_te = train_valid_test_split(
                X, Y, z, 0.34, 0.33
            )
            D_train = np.concatenate([X_tr, Y_tr], axis=1)
            D_test = np.concatenate([X_te, Y_te], axis=1)

            pca = PCA(n_components=1).fit(D_train)
            z_pred = pca.transform(D_test)

            tau, c = comp_ccorr(z_te, z_pred[:, 0])
            maxc = get_maxes(tau, c)[1]

            df = save_results(fname=interim / 'pca_res.csv', r=[maxc], N=1, method='PCA', dataset='logmap')

            self.assertTrue((interim / 'pca_res.csv').is_file())
            self.assertEqual(len(df), 1)
            self.assertGreater(maxc, -1.0)
            self.assertLess(maxc, 1.01)


class TestSmokeTentmap(unittest.TestCase):
    """End-to-end smoke test: tentmap PCA baseline."""

    def test_tentmap_pca_single_realization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            interim = Path(tmpdir) / 'interim'
            interim.mkdir()

            param_dict = dict(
                N=1, n=200, aint=(2, 10),
                A0=np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=float),
            )
            datasets, _params = gen_tentmapdata(param_dict)
            data = datasets[0]

            d_embed = 2
            X = time_delay_embedding(data[:, 1], delay=1, dimension=d_embed)
            Y = time_delay_embedding(data[:, 2], delay=1, dimension=d_embed)
            z = data[d_embed - 1:, 0]

            X_tr, Y_tr, z_tr, X_va, Y_va, z_va, X_te, Y_te, z_te = train_valid_test_split(
                X, Y, z, 0.34, 0.33
            )
            D_train = np.concatenate([X_tr, Y_tr], axis=1)
            D_test = np.concatenate([X_te, Y_te], axis=1)

            pca = PCA(n_components=1).fit(D_train)
            z_pred = pca.transform(D_test)

            maxc = get_maxes(*comp_ccorr(z_te, z_pred[:, 0]))[1]

            df = save_results(fname=interim / 'pca_res.csv', r=[maxc], N=1, method='PCA', dataset='tentmap')

            self.assertTrue((interim / 'pca_res.csv').is_file())
            self.assertGreater(maxc, -1.0)


class TestSmokeMaCo(unittest.TestCase):
    """End-to-end smoke test: MaCo training on tiny logmap data."""

    def test_maco_trains_without_error(self):
        param_dict = dict(
            N=1, n=300, rint=(3.8, 4.0),
            A0=np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=float),
            A=np.array([[1.0, 0.0, 0.0], [0.3, 1.0, 0.0], [0.4, 0.0, 1.0]], dtype=float),
        )
        datasets, _params = gen_logmapdata(param_dict)
        data = datasets[0].astype(float)

        device = torch.device('cpu')
        train_loader, test_loader, valid_loader, z_test = build_series_loaders(
            data, batch_size=50, trainset_size=50, testset_size=25, validset_size=25,
        )

        model_factory = lambda: MaCo(
            Ex=1, Ey=2, Ez=1,
            mh_kwargs=dict(n_h1=8, n_h2=8),
            ch_kwargs=dict(n_h1=8, n_out=1),
            preprocess_kwargs=dict(tau=1),
            device=device,
        )
        models, train_losses, valid_losses, best_model = train_and_select_best_model(
            model_factory, train_loader, valid_loader, n_models=2, n_epochs=3, lr=1e-2,
            disable_tqdm=True,
        )

        self.assertEqual(len(models), 2)
        self.assertEqual(len(train_losses), 3)  # 3 epochs -> 3 rows (n_epochs)
        self.assertIsNotNone(best_model)

        # Test loop should return a finite loss
        test_loss = best_model.test_loop(test_loader)
        self.assertGreater(test_loss, 0)
        self.assertFalse(np.isnan(test_loss))
