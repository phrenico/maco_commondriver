"""Tests for Logistic Map generator: LogMap, LogmapExpRunner, gen_logmapdata."""
import unittest

import numpy as np

from cdriver.datagen.logmap import LogMap, LogmapExpRunner, gen_logmapdata


class TestLogMap(unittest.TestCase):
    def setUp(self):
        self.r = np.array([3.9, 3.85, 3.8])
        self.A = np.array([[1.0, 0.0, 0.0], [0.3, 1.0, 0.0], [0.4, 0.0, 1.0]])
        self.x0 = np.array([0.5, 0.3, 0.7])
        self.lm = LogMap(self.r, self.A, self.x0)

    def test_bounder_keeps_values_in_range(self):
        x = np.array([1.2, -0.3, 0.5])
        bounded = self.lm.bounder(x)
        self.assertTrue(np.all(bounded >= 0))
        self.assertTrue(np.all(bounded <= 1))

    def test_bounder_does_not_change_valid_values(self):
        x = np.array([0.5, 0.8, 0.1])
        bounded = self.lm.bounder(x)
        np.testing.assert_array_almost_equal(bounded, x)

    def test_step_produces_correct_shape(self):
        result = self.lm.step(self.x0)
        self.assertEqual(result.shape, (3,))
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 1))

    def test_gen_dataset_shape(self):
        n = 50
        data = self.lm.gen_dataset(n)
        self.assertEqual(data.shape, (n, 3))


class TestLogmapExpRunner(unittest.TestCase):
    def test_reproducibility_with_seed(self):
        baseA = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=float)
        runner = LogmapExpRunner(nvars=3, baseA=baseA, r_interval=(3.8, 4.0))
        data1, _ = runner.gen_experiment(n=30, seed=42)
        data2, _ = runner.gen_experiment(n=30, seed=42)
        np.testing.assert_array_equal(data1, data2)

    def test_different_seeds_produce_different_data(self):
        baseA = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=float)
        runner = LogmapExpRunner(nvars=3, baseA=baseA, r_interval=(3.8, 4.0))
        data1, _ = runner.gen_experiment(n=30, seed=0)
        data2, _ = runner.gen_experiment(n=30, seed=1)
        self.assertFalse(np.allclose(data1, data2))

    def test_returned_params_have_expected_keys(self):
        baseA = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=float)
        runner = LogmapExpRunner(nvars=3, baseA=baseA, r_interval=(3.8, 4.0))
        _, params = runner.gen_experiment(n=30, seed=123)
        self.assertIn('r', params)
        self.assertIn('A', params)
        self.assertIn('x0', params)


class TestGenLogmapdata(unittest.TestCase):
    def test_output_sizes(self):
        param_dict = dict(
            N=3, n=100, rint=(3.8, 4.0),
            A0=np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=float),
            A=np.array([[1.0, 0.0, 0.0], [0.3, 1.0, 0.0], [0.4, 0.0, 1.0]], dtype=float),
        )
        datasets, params = gen_logmapdata(param_dict)
        self.assertEqual(len(datasets), 3)
        self.assertEqual(len(params), 3)
        self.assertEqual(datasets[0].shape, (100, 3))

    def test_seed_reproducibility(self):
        param_dict = dict(
            N=2, n=50, rint=(3.8, 4.0), seed=42,
            A0=np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=float),
            A=np.array([[1.0, 0.0, 0.0], [0.3, 1.0, 0.0], [0.4, 0.0, 1.0]], dtype=float),
        )
        d1, _ = gen_logmapdata(param_dict)
        d2, _ = gen_logmapdata(param_dict)
        np.testing.assert_array_equal(d1[0], d2[0])
