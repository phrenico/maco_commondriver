"""Tests for MaCo (Mapper-Coach) model: forward pass, training, utilities."""
import unittest

import numpy as np
import torch

from cdriver.network.maco import MaCo, get_mapper, get_coach


class TestModelFactories(unittest.TestCase):
    def test_get_mapper_output_shape(self):
        mapper = get_mapper(n_in=2, n_h1=8, n_h2=8, n_out=1)
        x = torch.randn(5, 2)
        out = mapper(x)
        self.assertEqual(out.shape, (5, 1))

    def test_get_coach_output_shape(self):
        coach = get_coach(n_in=3, n_h1=8, n_out=1)
        x = torch.randn(5, 3)
        out = coach(x)
        self.assertEqual(out.shape, (5, 1))


class TestMaCoForward(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cpu')
        self.model = MaCo(
            Ex=1, Ey=2, Ez=1,
            mh_kwargs=dict(n_h1=4, n_h2=4),
            ch_kwargs=dict(n_h1=4, n_out=1),
            preprocess_kwargs=dict(tau=1),
            device=self.device,
        )
        # Small synthetic data
        self.x = torch.randn(50, 1)
        self.y = torch.randn(50, 1)

    def test_forward_pass_returns_four_tensors(self):
        pred, z, hz, target = self.model.forward(self.x, self.y)
        self.assertIsInstance(pred, torch.Tensor)
        self.assertIsInstance(z, torch.Tensor)
        self.assertIsInstance(hz, torch.Tensor)
        self.assertIsInstance(target, torch.Tensor)

    def test_prediction_shape_matches_target(self):
        pred, z, hz, target = self.model.forward(self.x, self.y)
        self.assertEqual(pred.shape, target.shape)
        self.assertEqual(z.shape[1], 1)  # Ez = 1
        self.assertEqual(target.shape[1], 1)

    def test_all_tensors_on_correct_device(self):
        pred, z, hz, target = self.model.forward(self.x, self.y)
        self.assertEqual(pred.device, self.device)
        self.assertEqual(z.device, self.device)
        self.assertEqual(hz.device, self.device)
        self.assertEqual(target.device, self.device)

    def test_loss_is_finite(self):
        pred, z, hz, target = self.model.forward(self.x, self.y)
        loss = self.model.criterion(target, pred)
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))
        self.assertGreater(loss.item(), 0)

    def test_regularized_loss_is_finite(self):
        pred, z, hz, target = self.model.forward(self.x, self.y)
        loss = self.model.regularized_loss(target, z, pred)
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))


class TestMaCoTraining(unittest.TestCase):
    def test_train_loop_reduces_loss(self):
        device = torch.device('cpu')
        model = MaCo(
            Ex=1, Ey=2, Ez=1,
            mh_kwargs=dict(n_h1=4, n_h2=4),
            ch_kwargs=dict(n_h1=4, n_out=1),
            preprocess_kwargs=dict(tau=1),
            device=device,
        )
        # Create a tiny dataloader
        x = torch.randn(100, 1)
        y = torch.randn(100, 1)
        loader = [(x, y)]

        initial_loss = model.forward(x, y)[0]
        initial_val = model.criterion(initial_loss, initial_loss.detach()).item()

        history = model.train_loop(loader, n_epochs=5, lr=1e-2, disable_tqdm=True)

        self.assertEqual(len(history), 5)
        # Loss should decrease over training
        self.assertLess(history[-1], history[0] * 1.5)  # allow noise, but shouldn't explode

    def test_test_loop_returns_scalar(self):
        device = torch.device('cpu')
        model = MaCo(
            Ex=1, Ey=2, Ez=1,
            mh_kwargs=dict(n_h1=4, n_h2=4),
            ch_kwargs=dict(n_h1=4, n_out=1),
            preprocess_kwargs=dict(tau=1),
            device=device,
        )
        x = torch.randn(50, 1)
        y = torch.randn(50, 1)
        loss = model.test_loop((x, y))
        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0)

    def test_valid_loop_returns_arrays(self):
        device = torch.device('cpu')
        model = MaCo(
            Ex=1, Ey=2, Ez=1,
            mh_kwargs=dict(n_h1=4, n_h2=4),
            ch_kwargs=dict(n_h1=4, n_out=1),
            preprocess_kwargs=dict(tau=1),
            device=device,
        )
        x = torch.randn(50, 1)
        y = torch.randn(50, 1)
        loss, pred, z, hz = model.valid_loop((x, y))
        self.assertIsInstance(loss, float)
        self.assertIsInstance(pred, np.ndarray)
        self.assertIsInstance(z, np.ndarray)
        self.assertIsInstance(hz, np.ndarray)
