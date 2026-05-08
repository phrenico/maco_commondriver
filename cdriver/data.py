"""Data loading and time-delay embedding utilities for the MaCo pipeline."""

from functools import partial

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from sklearn.preprocessing import scale


def make_tde(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a time-delay embedding (TDE) from two 1-D time series.

    Parameters
    ----------
    x : torch.Tensor, shape ``(T, 1)``
        Target time series.
    y : torch.Tensor, shape ``(T, 1)``
        Driver-proxy time series.

    Returns
    -------
    Q : torch.Tensor, shape ``(T-1, 3)``
        Embedding vectors ``[x(t), y(t), y(t+1)]``.
    target : torch.Tensor, shape ``(T-1, 1)``
        Next-step target ``x(t+1)``.
    """
    Q = torch.cat((x[:-1], y[:-1], y[1:]), dim=-1)
    target = x[1:]
    return Q, target


def split_sets(
    data: list,
    trainset_size: float,
    testset_size: float,
    validset_size: float,
) -> list:
    """Split a list of tensors/arrays into train / test / validation subsets.

    Sizes are interpreted as proportions and normalised internally.
    """

    def _split_one(X, train_frac, test_frac, _valid_frac):
        N = X.shape[0]
        i_train = int(train_frac * N)
        i_test = int((train_frac + test_frac) * N)
        return X[:i_train], X[i_train:i_test], X[i_test:]

    S = trainset_size + testset_size + validset_size
    fracs = (trainset_size / S, testset_size / S, validset_size / S)
    splits = [_split_one(item, *fracs) for item in data]
    return list(zip(*splits))


def make_batches(
    Q: torch.Tensor, target: torch.Tensor, batch_size: int
) -> list:
    """Shuffle *Q* / *target* and return a list of ``(q_batch, t_batch)`` tuples."""
    indices = np.arange(Q.shape[0])
    np.random.shuffle(indices)
    return list(
        zip(Q[indices].split(batch_size), target[indices].split(batch_size))
    )


def load_data(
    csv_path: str,
    batch_size: int = 2000,
    trainset_size: float = 80,
    testset_size: float = 10,
    validset_size: float = 10,
) -> tuple:
    """Load the sample data CSV and return data-loaders.

    Parameters
    ----------
    csv_path : str
        Path to the sample data CSV (first column is index, columns 1 and 2 are
        *x* and *y* time series respectively).
    batch_size : int
        Mini-batch size for the training loader.
    trainset_size, testset_size, validset_size : float
        Relative sizes of the three splits (normalised internally).

    Returns
    -------
    train_loader : list of ``(Q_batch, target_batch)`` tuples
    test_loader  : tuple ``(Q_test, target_test)``
    valid_loader : tuple ``(Q_valid, target_valid)``
    z_valid      : np.ndarray – ground-truth common-driver for the validation set
    """
    data = pd.read_csv(csv_path, index_col=0).values

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.3,)),
            torch.Tensor.float,
            partial(torch.squeeze, dim=0),
        ]
    )

    x = transform(data[:, 1:2])
    y = transform(data[:, 2:3])
    # Ground-truth common driver (used only for validation evaluation)
    z = scale(data[:-1, 0])

    Q, target = make_tde(x, y)

    (Q_train, tgt_train, _), (Q_test, tgt_test, _), (Q_valid, tgt_valid, z_valid) = \
        split_sets([Q, target, torch.tensor(z).float().unsqueeze(1)],
                   trainset_size, testset_size, validset_size)

    train_loader = make_batches(Q_train, tgt_train, batch_size)
    return train_loader, (Q_test, tgt_test), (Q_valid, tgt_valid), z_valid
