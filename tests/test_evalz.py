"""Tests for evaluation metrics: comp_ccorr, get_maxes, eval_lin."""
import unittest

import numpy as np

from cdriver.evaluate.evalz import comp_ccorr, get_maxes, eval_lin


class TestCompCorr(unittest.TestCase):
    def test_identical_signals_produce_max_at_lag_zero(self):
        y = np.sin(np.linspace(0, 20 * np.pi, 500))
        tau, c = comp_ccorr(y, y)
        lag, maxval = get_maxes(tau, c)
        self.assertEqual(lag, 0)
        self.assertGreater(maxval, 0.99)

    def test_delayed_copy_produces_correct_lag(self):
        y = np.random.randn(200)
        delay = 5
        y_delayed = np.roll(y, delay)
        tau, c = comp_ccorr(y, y_delayed)
        lag, maxval = get_maxes(tau, c)
        # Should detect peak at either +delay or -delay depending on sign convention
        self.assertIn(abs(lag), (delay,))

    def test_orthogonal_signals_have_low_correlation(self):
        np.random.seed(0)
        y1 = np.random.randn(300)
        y2 = np.random.randn(300)
        tau, c = comp_ccorr(y1, y2)
        _lag, maxval = get_maxes(tau, c)
        self.assertLess(maxval, 0.5)

    def test_output_shapes(self):
        y = np.random.randn(100)
        tau, c = comp_ccorr(y, y)
        self.assertEqual(len(tau), 199)  # 2*T - 1
        self.assertEqual(len(c), 199)


class TestEvalLin(unittest.TestCase):
    def test_perfect_linear_relationship(self):
        X = np.linspace(0, 10, 100).reshape(-1, 1)
        Y = 3 * X + 2
        regmod, regmod2 = eval_lin(X, Y)
        self.assertGreater(regmod.score(X, Y), 0.999)
        self.assertGreater(regmod2.score(Y, X), 0.999)

    def test_returns_two_models(self):
        X = np.random.randn(50, 2)
        Y = np.random.randn(50, 1)
        regmod, regmod2 = eval_lin(X, Y)
        self.assertTrue(hasattr(regmod, 'coef_'))
        self.assertTrue(hasattr(regmod2, 'coef_'))
