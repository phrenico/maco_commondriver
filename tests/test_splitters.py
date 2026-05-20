"""Tests for data splitters: train_test_split, train_valid_test_split."""
import unittest

import numpy as np

from cdriver.preprocessing.splitters import train_test_split, train_valid_test_split


class TestTrainTestSplit(unittest.TestCase):
    def test_output_sizes(self):
        N = 100
        X = np.random.randn(N, 3)
        Y = np.random.randn(N, 2)
        z = np.random.randn(N)
        X_tr, Y_tr, z_tr, X_te, Y_te, z_te = train_test_split(X, Y, z, train_size=0.8)
        self.assertEqual(len(X_tr), 80)
        self.assertEqual(len(X_te), 20)
        self.assertEqual(len(z_tr), 80)
        self.assertEqual(len(z_te), 20)

    def test_train_all(self):
        X = np.random.randn(50, 1)
        X_tr, Y_tr, z_tr, X_te, Y_te, z_te = train_test_split(X, X, X, train_size=1.0)
        self.assertEqual(len(X_tr), 50)
        self.assertEqual(len(X_te), 0)

    def test_small_split(self):
        X = np.random.randn(10, 2)
        X_tr, Y_tr, z_tr, X_te, Y_te, z_te = train_test_split(X, X, X.squeeze(), train_size=0.1)
        self.assertEqual(len(X_tr), 1)
        self.assertEqual(len(X_te), 9)


class TestTrainValidTestSplit(unittest.TestCase):
    def test_output_sizes(self):
        N = 100
        X = np.random.randn(N, 3)
        Y = np.random.randn(N, 2)
        z = np.random.randn(N)
        parts = train_valid_test_split(X, Y, z, train_size=0.6, valid_size=0.2)
        X_tr, Y_tr, z_tr, X_va, Y_va, z_va, X_te, Y_te, z_te = parts
        self.assertEqual(len(X_tr), 60)
        self.assertEqual(len(X_va), 20)
        self.assertEqual(len(X_te), 20)
        self.assertEqual(len(z_tr), 60)
        self.assertEqual(len(z_va), 20)
        self.assertEqual(len(z_te), 20)

    def test_contiguous_no_overlap(self):
        N = 50
        X = np.arange(N).reshape(-1, 1).astype(float)
        parts = train_valid_test_split(X, X.copy(), X.squeeze().copy(), 0.5, 0.3)
        X_tr, _, _, X_va, _, _, X_te, _, _ = parts
        # Check no index overlap
        tr_set = set(X_tr.flatten().astype(int))
        va_set = set(X_va.flatten().astype(int))
        te_set = set(X_te.flatten().astype(int))
        self.assertTrue(tr_set.isdisjoint(va_set))
        self.assertTrue(tr_set.isdisjoint(te_set))
        self.assertTrue(va_set.isdisjoint(te_set))
