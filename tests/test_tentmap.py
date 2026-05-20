"""Tests for Tent Map generator: TentMap, TentMapExpRunner, gen_tentmapdata."""
import unittest

import numpy as np

from cdriver.datagen.tent_map import TentMap, TentMapExpRunner, gen_tentmapdata


class TestTentMap(unittest.TestCase):
    def setUp(self):
        self.alpha = np.array([3.0, 4.0, 5.0])
        self.A = np.array([[1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.5, 0.0, 1.0]])
        self.x0 = np.array([0.5, 0.3, 0.7])
        self.tm = TentMap(self.alpha, self.A, self.x0)

    def test_bounder_keeps_values_in_range(self):
        x = np.array([1.5, -0.2, 0.5])
        bounded = self.tm.bounder(x)
        self.assertTrue(np.all(bounded >= 0))
        self.assertTrue(np.all(bounded <= 1))

    def test_step_produces_correct_shape(self):
        result = self.tm.step(self.x0)
        self.assertEqual(result.shape, (3,))
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 1))

    def test_gen_dataset_shape(self):
        n = 60
        data = self.tm.gen_dataset(n)
        self.assertEqual(data.shape, (n, 3))


class TestTentMapExpRunner(unittest.TestCase):
    def test_reproducibility_with_seed(self):
        baseA = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=float)
        runner = TentMapExpRunner(nvars=3, baseA=baseA, a_interval=(2, 10))
        data1, _ = runner.gen_experiment(n=30, seed=42)
        data2, _ = runner.gen_experiment(n=30, seed=42)
        np.testing.assert_array_equal(data1, data2)

    def test_returned_params_have_expected_keys(self):
        baseA = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=float)
        runner = TentMapExpRunner(nvars=3, baseA=baseA, a_interval=(2, 10))
        _, params = runner.gen_experiment(n=30, seed=123)
        self.assertIn('a', params)
        self.assertIn('A', params)
        self.assertIn('x0', params)


class TestGenTentmapdata(unittest.TestCase):
    def test_output_sizes(self):
        param_dict = dict(
            N=3, n=100, aint=(2, 10),
            A0=np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=float),
        )
        datasets, params = gen_tentmapdata(param_dict)
        self.assertEqual(len(datasets), 3)
        self.assertEqual(len(params), 3)
        self.assertEqual(datasets[0].shape, (100, 3))

    def test_seed_reproducibility(self):
        param_dict = dict(
            N=2, n=50, aint=(2, 10), seed=42,
            A0=np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=float),
        )
        d1, _ = gen_tentmapdata(param_dict)
        d2, _ = gen_tentmapdata(param_dict)
        np.testing.assert_array_equal(d1[0], d2[0])
