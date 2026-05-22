import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from cdriver.evaluate.evalz import comp_ccorr, get_maxes


def split_sets(x, y, z, trainset_size, testset_size, validset_size):
    """Split aligned arrays by percentage."""
    n = x.shape[0]
    n_trainset = int(trainset_size * n / 100)
    n_validset = int(validset_size * n / 100)

    x_trainset = x[:n_trainset]
    x_validset = x[n_trainset:n_trainset + n_validset]
    x_testset = x[n_trainset + n_validset:]

    y_trainset = y[:n_trainset]
    y_validset = y[n_trainset:n_trainset + n_validset]
    y_testset = y[n_trainset + n_validset:]

    z_trainset = z[:n_trainset]
    z_validset = z[n_trainset:n_trainset + n_validset]
    z_testset = z[n_trainset + n_validset:]
    return ((x_trainset, y_trainset, z_trainset),
            (x_testset, y_testset, z_testset),
            (x_validset, y_validset, z_validset))


def build_loaders(x, y, z, batch_size, trainset_size=50, testset_size=50, validset_size=0, transform=None):
    """Create train/test/validation loaders for MaCo experiments."""
    if transform is None:
        transform = transforms.ToTensor()

    ((x_train, y_train, _),
     (x_test, y_test, z_test),
     (x_valid, y_valid, _)) = split_sets(x, y, z, trainset_size, testset_size, validset_size)

    train_dataset = TensorDataset(transform(x_train), transform(y_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    test_loader = transform(x_test), transform(y_test)
    valid_loader = transform(x_valid), transform(y_valid)
    return train_loader, test_loader, valid_loader, z_test


def build_series_loaders(data, batch_size, trainset_size=50, testset_size=50, validset_size=0,
                         x_slice=slice(1, 2), y_slice=slice(2, 3), z_index=0, transform=None):
    """Extract observed series and create loaders for a dataset matrix."""
    x = data[:, x_slice]
    y = data[:, y_slice]
    z = data[:-1, z_index]
    return build_loaders(x, y, z, batch_size, trainset_size, testset_size, validset_size, transform=transform)


def train_and_select_best_model(model_factory, train_loader, valid_loader, n_models, n_epochs, lr,
                                disable_tqdm=True, leave=False, desc=None, configure_model=None):
    """Train several MaCo models and return the best one by validation loss."""
    models = []
    for _ in range(n_models):
        model = model_factory()
        if configure_model is not None:
            configure_model(model)
        models.append(model)

    train_losses = []
    valid_losses = []
    for model_index in tqdm(range(n_models), disable=disable_tqdm, leave=leave, desc=desc):
        train_losses.append(models[model_index].train_loop(train_loader,
                                                           n_epochs,
                                                           lr=lr,
                                                           disable_tqdm=True))
        valid_losses.append(models[model_index].test_loop(valid_loader))

    train_losses = np.array(train_losses).T if train_losses else np.array([])
    valid_losses = np.array(valid_losses)
    best_model = models[int(np.argmin(valid_losses))]
    return models, train_losses, valid_losses, best_model


def score_latent_reconstruction(z_pred, z_test):
    """Score the learned latent against the hidden signal with max absolute cross-correlation."""
    tau, c = comp_ccorr(z_pred, z_test)
    return get_maxes(tau, c)[1]


def create_sweep_df(maxdict):
    """Create a tidy dataframe from a sweep keyed by parameter value."""
    frames = [pd.DataFrame({'r2': values, 'L': len(values) * [key]}) for key, values in maxdict.items()]
    return pd.concat(frames, axis=0)


def get_default_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")