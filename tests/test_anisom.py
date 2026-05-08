"""Tests for the MaCo model (anisometric reconstruction)."""

import numpy as np
import pytest
import torch

from cdriver.model import MaCo, get_mapper, get_coach


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def default_device():
    return torch.device("cpu")


@pytest.fixture()
def tiny_maco(default_device):
    """Return a small MaCo instance suitable for fast unit tests."""
    return MaCo(
        Ex=1,
        Ey=2,
        Ez=1,
        mh_kwargs=dict(n_h1=4, n_h2=4),
        ch_kwargs=dict(n_h1=4),
        device=default_device,
    )


@pytest.fixture()
def small_batch():
    """Return a (Q, target) mini-batch of 32 samples."""
    torch.manual_seed(0)
    Q = torch.randn(32, 3)       # Ex=1, Ey=2  →  1+2 = 3 input dims
    target = torch.randn(32, 1)
    return Q, target


@pytest.fixture()
def small_loader(small_batch):
    """Return a list of one-batch training loader."""
    return [small_batch]


# ---------------------------------------------------------------------------
# Network architecture helpers
# ---------------------------------------------------------------------------

class TestGetMapper:
    def test_output_shape(self):
        net = get_mapper(n_in=4, n_h1=8, n_h2=8, n_out=2)
        x = torch.randn(16, 4)
        out = net(x)
        assert out.shape == (16, 2), f"Unexpected mapper output shape: {out.shape}"


class TestGetCoach:
    def test_output_shape(self):
        net = get_coach(n_in=3, n_h1=8, n_out=1)
        x = torch.randn(16, 3)
        out = net(x)
        assert out.shape == (16, 1), f"Unexpected coach output shape: {out.shape}"


# ---------------------------------------------------------------------------
# MaCo forward pass (anisometric reconstruction)
# ---------------------------------------------------------------------------

class TestMaCoForward:
    def test_output_shapes(self, tiny_maco, small_batch):
        Q, _ = small_batch
        pred, z, hz = tiny_maco(Q)

        assert pred.shape == (32, 1), f"pred shape mismatch: {pred.shape}"
        assert z.shape == (32, 1),    f"z shape mismatch: {z.shape}"
        assert hz.shape == (32, 1),   f"hz shape mismatch: {hz.shape}"

    def test_hz_is_nonnegative(self, tiny_maco, small_batch):
        """Post-activation latent hz = relu(z) must be ≥ 0."""
        Q, _ = small_batch
        _, _z, hz = tiny_maco(Q)
        assert (hz >= 0).all(), "hz should be non-negative (relu output)"

    def test_deterministic_forward(self, tiny_maco, small_batch):
        """Two forward passes with the same input must produce identical output."""
        Q, _ = small_batch
        tiny_maco.eval()
        with torch.no_grad():
            pred1, z1, hz1 = tiny_maco(Q)
            pred2, z2, hz2 = tiny_maco(Q)
        assert torch.allclose(pred1, pred2)
        assert torch.allclose(z1, z2)


# ---------------------------------------------------------------------------
# MaCo training loop (headless – no plots opened)
# ---------------------------------------------------------------------------

class TestMaCoTraining:
    def test_loss_decreases(self, tiny_maco, small_loader):
        """Training for a few epochs should reduce the loss."""
        history = tiny_maco.train_loop(small_loader, n_epochs=10, lr=1e-2)
        assert len(history) == 10, "History length should equal n_epochs"
        assert history[-1] < history[0], (
            "Loss did not decrease after 10 epochs – model may not be training"
        )

    def test_train_loss_history_accumulates(self, tiny_maco, small_loader):
        tiny_maco.train_loop(small_loader, n_epochs=3, lr=1e-2)
        tiny_maco.train_loop(small_loader, n_epochs=3, lr=1e-2)
        assert len(tiny_maco.train_loss_history) == 6


# ---------------------------------------------------------------------------
# MaCo valid_loop – reconstruction quality
# ---------------------------------------------------------------------------

class TestMaCoValidLoop:
    def test_valid_loop_output_shapes(self, tiny_maco, small_batch):
        loss, x_pred, z_pred, hz_pred = tiny_maco.valid_loop(small_batch)

        assert isinstance(loss, float), "loss should be a Python float"
        assert x_pred.shape == (32,),   f"x_pred shape mismatch: {x_pred.shape}"
        assert z_pred.shape == (32,),   f"z_pred shape mismatch: {z_pred.shape}"
        assert hz_pred.shape == (32,),  f"hz_pred shape mismatch: {hz_pred.shape}"

    def test_valid_loss_is_finite(self, tiny_maco, small_batch):
        loss, _, _, _ = tiny_maco.valid_loop(small_batch)
        assert np.isfinite(loss), f"Validation loss is not finite: {loss}"
