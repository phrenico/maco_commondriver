"""MaCo (Mapper-Coach) neural-network model for common-driver identification."""

import numpy as np
import torch
from torch.nn import Module, Sequential, Linear, ReLU, MSELoss
from torch.nn.functional import relu
import torch.optim as optim
from tqdm import tqdm


def get_mapper(n_in, n_h1, n_h2, n_out):
    """Return a 3-layer mapper network."""
    return Sequential(
        Linear(n_in, n_h1),
        ReLU(),
        Linear(n_h1, n_h2),
        ReLU(),
        Linear(n_h2, n_out),
    )


def get_coach(n_in, n_h1, n_out):
    """Return a 2-layer coach (predictor) network."""
    return Sequential(
        Linear(n_in, n_h1),
        ReLU(),
        Linear(n_h1, n_out),
    )


class MaCo(Module):
    """Mapper-Coach model.

    Parameters
    ----------
    Ex : int
        Dimension of the *x* (target) embedding.
    Ey : int
        Dimension of the *y* (driver proxy) embedding.
    Ez : int
        Dimension of the latent (common-driver) representation.
    mh_kwargs : dict
        Extra keyword arguments forwarded to :func:`get_mapper` (``n_h1``, ``n_h2``).
    ch_kwargs : dict
        Extra keyword arguments forwarded to :func:`get_coach` (``n_h1``).
    device : torch.device
        Device on which the model lives.
    """

    def __init__(self, Ex, Ey, Ez, mh_kwargs, ch_kwargs, device):
        super().__init__()
        self.mapper = get_mapper(n_in=Ey, n_out=Ez, **mh_kwargs)
        self.coach_x = get_coach(Ex + Ez, n_out=1, **ch_kwargs)

        self.Ex = Ex
        self.Ey = Ey
        self.Ez = Ez
        self.mh_params = mh_kwargs
        self.ch_params = ch_kwargs
        self.train_loss_history = []
        self.criterion = MSELoss()
        self.device = device

    def forward(self, q):
        """Forward pass.

        Parameters
        ----------
        q : torch.Tensor, shape ``(batch, Ex + Ey + Ey)``

        Returns
        -------
        pred : torch.Tensor, shape ``(batch, 1)``
        z    : torch.Tensor, shape ``(batch, Ez)``   – pre-activation latent
        hz   : torch.Tensor, shape ``(batch, Ez)``   – post-activation latent
        """
        z = self.mapper(q[:, self.Ex : self.Ex + self.Ey])
        hz = relu(z)
        mx = torch.cat((hz, q[:, : self.Ex]), dim=1)
        pred = self.coach_x(mx)
        return pred, z, hz

    def train_loop(self, loader, n_epochs, lr=1e-2):
        """Train for *n_epochs* over *loader* and return loss history."""
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        for _epoch in tqdm(range(n_epochs), leave=False):
            losses = []
            for q, target in loader:
                self.optimizer.zero_grad()
                pred, _z, _hz = self.forward(q.to(self.device))
                loss = self.criterion(target.to(self.device), pred)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
            self.train_loss_history.append(np.mean(losses))
        return list(self.train_loss_history)

    def test_loop(self, loader):
        """Return scalar test loss on *loader* (a ``(Q, target)`` tuple)."""
        q, target = loader
        pred, _z, _hz = self.forward(q)
        return self.criterion(target, pred).item()

    def valid_loop(self, loader):
        """Evaluate on *loader* and return ``(loss, x_pred, z_pred, hz_pred)``."""
        q, target = loader
        pred, z, hz = self.forward(q)
        loss = self.criterion(target, pred).item()
        return (
            loss,
            pred.squeeze().detach().numpy(),
            z.squeeze().detach().numpy(),
            hz.squeeze().detach().numpy(),
        )
