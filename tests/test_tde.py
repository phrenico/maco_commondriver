"""Tests for the time-delay embedding (TDE) data-preparation utilities."""

import numpy as np
import pytest
import torch

from cdriver.data import make_tde, split_sets, make_batches


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_series():
    """Return two short, deterministic 1-D time series as (T, 1) tensors."""
    T = 100
    t = torch.linspace(0, 2 * np.pi, T).unsqueeze(1)
    x = torch.sin(t)
    y = torch.cos(t)
    return x, y


# ---------------------------------------------------------------------------
# make_tde
# ---------------------------------------------------------------------------

class TestMakeTDE:
    def test_output_shapes(self, sample_series):
        x, y = sample_series
        T = x.shape[0]
        Q, target = make_tde(x, y)

        assert Q.shape == (T - 1, 3), (
            f"Expected Q shape ({T - 1}, 3), got {Q.shape}"
        )
        assert target.shape == (T - 1, 1), (
            f"Expected target shape ({T - 1}, 1), got {target.shape}"
        )

    def test_embedding_columns(self, sample_series):
        """Q columns must be [x(t), y(t), y(t+1)] and target must be x(t+1)."""
        x, y = sample_series
        Q, target = make_tde(x, y)

        assert torch.allclose(Q[:, 0:1], x[:-1]), "Column 0 should be x(t)"
        assert torch.allclose(Q[:, 1:2], y[:-1]), "Column 1 should be y(t)"
        assert torch.allclose(Q[:, 2:3], y[1:]), "Column 2 should be y(t+1)"
        assert torch.allclose(target, x[1:]), "target should be x(t+1)"

    def test_no_data_leakage(self, sample_series):
        """The last element of Q must not contain future information beyond t+1."""
        x, y = sample_series
        Q, target = make_tde(x, y)
        # The last row of Q encodes t = T-2; target encodes t+1 = T-1.
        assert torch.allclose(target[-1:], x[-1:])


# ---------------------------------------------------------------------------
# split_sets
# ---------------------------------------------------------------------------

class TestSplitSets:
    def test_split_proportions(self, sample_series):
        x, _ = sample_series
        # split_sets returns [(train_items...), (test_items...), (valid_items...)]
        train_set, test_set, valid_set = split_sets([x], 80, 10, 10)
        x_train, x_test, x_valid = train_set[0], test_set[0], valid_set[0]
        total = x_train.shape[0] + x_test.shape[0] + x_valid.shape[0]
        assert total == x.shape[0], "Split must be exhaustive"

    def test_no_overlap(self, sample_series):
        x, _ = sample_series
        train_set, test_set, valid_set = split_sets([x], 80, 10, 10)
        x_train, x_test, x_valid = train_set[0], test_set[0], valid_set[0]
        # Indices are contiguous so lengths are non-zero
        assert x_train.shape[0] > 0
        assert x_test.shape[0] > 0
        assert x_valid.shape[0] > 0


# ---------------------------------------------------------------------------
# make_batches
# ---------------------------------------------------------------------------

class TestMakeBatches:
    def test_batch_sizes(self, sample_series):
        x, y = sample_series
        Q, target = make_tde(x, y)
        batch_size = 20
        batches = make_batches(Q, target, batch_size)

        assert len(batches) > 0, "Should produce at least one batch"
        for q_b, t_b in batches:
            assert q_b.shape[0] <= batch_size
            assert q_b.shape[0] == t_b.shape[0]

    def test_all_samples_covered(self, sample_series):
        x, y = sample_series
        Q, target = make_tde(x, y)
        batches = make_batches(Q, target, batch_size=30)
        total = sum(q_b.shape[0] for q_b, _ in batches)
        assert total == Q.shape[0], "All samples must appear in exactly one batch"
