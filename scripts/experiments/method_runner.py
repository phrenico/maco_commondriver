"""Parameterized method runner for experiment families.

Replaces ~30 near-identical per-method experiment scripts with a single CLI
entry point that accepts --family, --method, and --config.

Usage (standalone):
    python -m scripts.experiments.method_runner --family lorenz --method pca [--config path/to/config.py]

Usage (from thin wrapper — preferred for registry compatibility):
    # thin wrapper calls:
    run_method('lorenz', 'pca', config_path)

See the EXPERIMENT_REGISTRY update section in dev/fix_plan.md for the
migration path from per-script entry points to registry-driven dispatch.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
from tqdm import tqdm

from cdriver.evaluate.evalz import comp_ccorr, get_maxes
from cdriver.preprocessing.splitters import train_valid_test_split
from cdriver.preprocessing.tde import time_delay_embedding
from cdriver.savers.saver import save_results
from scripts.experiments.config_loader import get_config, resolve_paths
from scripts.experiments.maco_utils import build_series_loaders, get_default_device, score_latent_reconstruction, train_and_select_best_model

_REPO_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Data loaders per family
# ---------------------------------------------------------------------------

def _load_lorenz(cfg: dict, n_iter: int) -> tuple:
    """Load one Lorenz realization and return (X_train, Y_train, z_train, X_test, Y_test, z_test)."""
    data_path_template = str(_REPO_ROOT / cfg['data']['data_path_template'])
    data = np.load(data_path_template.format(n_iter))
    X = data['v'][:, 3:6]
    Y = data['v'][:, 6:]
    z = data['v'][:, 1]
    train_split = cfg['preprocessing']['train_split']
    valid_split = cfg['preprocessing']['valid_split']
    return train_valid_test_split(X, Y, z, train_split, valid_split)


def _load_logmap(cfg: dict, n_iter: int, dataset: list, d_embed: int | None = None):
    """Load one logmap realization with TDE."""
    if d_embed is None:
        d_embed = cfg['preprocessing'].get('d_embed', 3)
    data = dataset[n_iter].astype(float)
    X = time_delay_embedding(data[:, 1], delay=1, dimension=d_embed)
    Y = time_delay_embedding(data[:, 2], delay=1, dimension=d_embed)
    z = data[d_embed - 1:, 0]
    train_split = cfg['preprocessing']['train_split']
    valid_split = cfg['preprocessing']['valid_split']
    return train_valid_test_split(X, Y, z, train_split, valid_split)


def _load_tentmap(cfg: dict, n_iter: int, dataset: list):
    """Load one tentmap realization with TDE."""
    d_embed = cfg['preprocessing'].get('d_embed', 2)
    return _load_logmap(cfg, n_iter, dataset, d_embed=d_embed)


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def _make_sklearn_model(model_class, **kwargs):
    """Factory for sklearn-style models: PCA, FastICA, KernelPCA, CCA."""
    return model_class(**kwargs)


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_baseline_method(
    family_key: str,
    method_name: str,
    model_factory: Callable[[dict], Any],
    data_loader: Callable[[dict, int], tuple] | None = None,
    config_path: str | None = None,
    *,
    dataset_key: str = 'datagen',
    result_file: str | None = None,
    use_concatenated: bool = True,
    n_components_from_model: bool = True,
) -> None:
    """Run a baseline method on all realizations of an experiment family.

    Parameters
    ----------
    family_key: 'logmaps', 'tentmaps', 'lorenz', etc.
    method_name: 'PCA', 'ICA', etc. (used for save_results labeling)
    model_factory: callable(method_config_dict) -> sklearn-style model
    data_loader: callable(cfg, n_iter) -> (X_train,Y_train,z_train, X_test,Y_test,z_test)
        If None, auto-selected based on family_key.
    config_path: optional --config override path
    dataset_key: 'datagen' or 'data' in config for N and dataset loading
    result_file: override for the output CSV name (default: {method_name.lower()}_res.csv)
    use_concatenated: if True, concatenate X and Y before fit/transform
    n_components_from_model: if True, use model.transform output shape for component count
    """
    cfg = resolve_paths(get_config(family_key, config_path), _REPO_ROOT)

    interim_res_path = cfg['paths']['interim_res_path']
    os.makedirs(interim_res_path, exist_ok=True)

    method_cfg = cfg['methods'][method_name.lower()]
    N = cfg.get('datagen', cfg.get('data', {}))['N']

    # Auto-select data loader
    if data_loader is None:
        if family_key in ('logmaps', 'example_logmap'):
            from cdriver.datagen.logmap import gen_logmapdata
            dataset, _params = gen_logmapdata(cfg['datagen'])
            data_loader = lambda cfg, i, ds=dataset: _load_logmap(cfg, i, ds)
        elif family_key == 'tentmaps':
            from cdriver.datagen.tent_map import gen_tentmapdata
            dataset, _params = gen_tentmapdata(cfg['datagen'])
            data_loader = lambda cfg, i, ds=dataset: _load_tentmap(cfg, i, ds)
        elif family_key == 'lorenz':
            data_loader = _load_lorenz
        else:
            raise ValueError(f'Unknown family: {family_key}')

    maxcs = []
    for n_iter in tqdm(range(N), desc=f'{method_name} ({family_key})'):
        X_train, Y_train, z_train, X_valid, Y_valid, z_valid, X_test, Y_test, z_test = data_loader(cfg, n_iter)

        if use_concatenated:
            D_train = np.concatenate([X_train, Y_train], axis=1)
            D_test = np.concatenate([X_test, Y_test], axis=1)
        else:
            D_train, D_test = X_train, X_test

        model = model_factory(method_cfg)
        model.fit(D_train)
        z_pred = model.transform(D_test)

        if n_components_from_model and z_pred.ndim > 1:
            n_comp = z_pred.shape[1]
            m = max(get_maxes(*comp_ccorr(z_test, z_pred[:, j]))[1] for j in range(n_comp))
        else:
            m = get_maxes(*comp_ccorr(z_test, z_pred))[1]
        maxcs.append(m)

    fname = result_file if result_file else f'{method_name.lower()}_res.csv'
    df = save_results(fname=interim_res_path / fname, r=maxcs, N=N, method=method_name, dataset=family_key)
    return df


def run_random_baseline(family_key: str, config_path: str | None = None):
    """Random baseline — uses np.random.rand or shuffle_phase."""
    cfg = resolve_paths(get_config(family_key, config_path), _REPO_ROOT)
    interim_res_path = cfg['paths']['interim_res_path']
    os.makedirs(interim_res_path, exist_ok=True)

    N = cfg.get('datagen', cfg.get('data', {}))['N']

    if family_key == 'lorenz':
        data_loader = _load_lorenz
        use_shuffle = True
    elif family_key == 'logmaps':
        from cdriver.datagen.logmap import gen_logmapdata
        dataset, _ = gen_logmapdata(cfg['datagen'])
        data_loader = lambda cfg, i, ds=dataset: _load_logmap(cfg, i, ds)
        use_shuffle = False
    elif family_key == 'tentmaps':
        from cdriver.datagen.tent_map import gen_tentmapdata
        dataset, _ = gen_tentmapdata(cfg['datagen'])
        data_loader = lambda cfg, i, ds=dataset: _load_tentmap(cfg, i, ds)
        use_shuffle = False
    else:
        raise ValueError(f'Unknown family: {family_key}')

    maxcs = []
    for n_iter in tqdm(range(N), desc=f'Random ({family_key})'):
        X_train, Y_train, z_train, X_valid, Y_valid, z_valid, X_test, Y_test, z_test = data_loader(cfg, n_iter)
        if use_shuffle:
            from cdriver.datagen.control import shuffle_phase
            z_pred = shuffle_phase(z_test)
        else:
            z_pred = np.random.rand(len(z_test))
        maxcs.append(get_maxes(*comp_ccorr(z_test, z_pred))[1])

    save_results(fname=interim_res_path / 'random_res.csv', r=maxcs, N=N, method='Random', dataset=family_key)


def run_dca_method(family_key: str, config_path: str | None = None):
    """DCA baseline — uses dca.DynamicalComponentsAnalysis (sklearn-compatible API)."""
    from dca import DynamicalComponentsAnalysis as DCA

    cfg = resolve_paths(get_config(family_key, config_path), _REPO_ROOT)
    interim_res_path = cfg['paths']['interim_res_path']
    os.makedirs(interim_res_path, exist_ok=True)

    method_cfg = cfg['methods']['dca']
    n_components = method_cfg['n_components']
    T = method_cfg['T']
    n_init = method_cfg['n_init']
    N = cfg.get('datagen', cfg.get('data', {}))['N']

    if family_key == 'lorenz':
        data_loader = _load_lorenz
    elif family_key == 'logmaps':
        from cdriver.datagen.logmap import gen_logmapdata
        dataset, _ = gen_logmapdata(cfg['datagen'])
        data_loader = lambda cfg, i, ds=dataset: _load_logmap(cfg, i, ds)
    elif family_key == 'tentmaps':
        from cdriver.datagen.tent_map import gen_tentmapdata
        dataset, _ = gen_tentmapdata(cfg['datagen'])
        data_loader = lambda cfg, i, ds=dataset: _load_tentmap(cfg, i, ds)
    else:
        raise ValueError(f'Unknown family: {family_key}')

    maxcs = []
    for n_iter in tqdm(range(N), desc=f'DCA ({family_key})'):
        X_train, Y_train, z_train, X_valid, Y_valid, z_valid, X_test, Y_test, z_test = data_loader(cfg, n_iter)
        D_train = np.concatenate([X_train, Y_train], axis=1)
        D_test = np.concatenate([X_test, Y_test], axis=1)

        model = DCA(d=n_components, T=T, n_init=n_init)
        model.fit(D_train)
        z_pred = model.transform(D_test)
        m = max(get_maxes(*comp_ccorr(z_test, z_pred[:, j]))[1] for j in range(n_components))
        maxcs.append(m)

    save_results(fname=interim_res_path / 'dca_res.csv', r=maxcs, N=N, method='DCA', dataset=family_key)


def run_sfa_method(family_key: str, config_path: str | None = None):
    """SFA baseline — uses sksfa.SFA with optional PolynomialFeatures."""
    import sksfa
    from sklearn.preprocessing import PolynomialFeatures

    cfg = resolve_paths(get_config(family_key, config_path), _REPO_ROOT)
    interim_res_path = cfg['paths']['interim_res_path']
    os.makedirs(interim_res_path, exist_ok=True)

    method_cfg = cfg['methods']['sfa']
    n_components = method_cfg['n_components']
    poly_degree = method_cfg.get('poly_degree')
    N = cfg.get('datagen', cfg.get('data', {}))['N']

    if family_key == 'lorenz':
        data_loader = _load_lorenz
    elif family_key == 'logmaps':
        from cdriver.datagen.logmap import gen_logmapdata
        dataset, _ = gen_logmapdata(cfg['datagen'])
        data_loader = lambda cfg, i, ds=dataset: _load_logmap(cfg, i, ds)
    elif family_key == 'tentmaps':
        from cdriver.datagen.tent_map import gen_tentmapdata
        dataset, _ = gen_tentmapdata(cfg['datagen'])
        data_loader = lambda cfg, i, ds=dataset: _load_tentmap(cfg, i, ds)
    else:
        raise ValueError(f'Unknown family: {family_key}')

    maxcs = []
    for n_iter in tqdm(range(N), desc=f'SFA ({family_key})'):
        X_train, Y_train, z_train, X_valid, Y_valid, z_valid, X_test, Y_test, z_test = data_loader(cfg, n_iter)

        if family_key != 'lorenz':
            D_train = np.concatenate([X_train, Y_train], axis=1)
            D_test = np.concatenate([X_test, Y_test], axis=1)
        else:
            D_train, D_test = X_train, X_test

        if poly_degree is not None:
            poly = PolynomialFeatures(degree=poly_degree)
            D_train = poly.fit_transform(D_train)
            D_test = poly.transform(D_test)

        sfa = sksfa.SFA(n_components=n_components)
        sfa.fit(D_train)
        z_pred = sfa.transform(D_test).squeeze()

        if z_pred.ndim == 2 and z_pred.shape[1] > 1:
            m = max(get_maxes(*comp_ccorr(z_test, z_pred[:, j]))[1] for j in range(z_pred.shape[1]))
        else:
            m = get_maxes(*comp_ccorr(z_pred, z_test))[1]
        maxcs.append(m)

    save_results(fname=interim_res_path / 'sfa_res.csv', r=maxcs, N=N, method='SFA', dataset=family_key)


def run_shrec_method(family_key: str, config_path: str | None = None):
    """ShRec baseline — uses shrec.models.RecurrenceManifold."""
    from shrec.models import RecurrenceManifold

    cfg = resolve_paths(get_config(family_key, config_path), _REPO_ROOT)
    interim_res_path = cfg['paths']['interim_res_path']
    os.makedirs(interim_res_path, exist_ok=True)

    d_embed = cfg['methods']['shrec']['d_embed']
    N = cfg.get('datagen', cfg.get('data', {}))['N']

    if family_key == 'lorenz':
        data_loader = _load_lorenz
        use_test = True
    elif family_key == 'logmaps':
        from cdriver.datagen.logmap import gen_logmapdata
        dataset, _ = gen_logmapdata(cfg['datagen'])
        def data_loader(cfg, i, ds=dataset):
            data = ds[i]
            X = data[:, 1:]
            y = data[:, 0]
            split = cfg['preprocessing']['train_split']
            vsplit = cfg['preprocessing']['valid_split']
            return train_valid_test_split(X, X, y, split, vsplit)
        use_test = False
    elif family_key == 'tentmaps':
        from cdriver.datagen.tent_map import gen_tentmapdata
        dataset, _ = gen_tentmapdata(cfg['datagen'])
        def data_loader(cfg, i, ds=dataset):
            data = ds[i]
            X = data[:, 1:]
            y = data[:, 0]
            split = cfg['preprocessing']['train_split']
            vsplit = cfg['preprocessing']['valid_split']
            return train_valid_test_split(X, X, y, split, vsplit)
        use_test = False
    else:
        raise ValueError(f'Unknown family: {family_key}')

    maxcs = []
    for n_iter in tqdm(range(N), desc=f'ShRec ({family_key})'):
        X_train, Y_train, z_train, X_valid, Y_valid, z_valid, X_test, Y_test, z_test = data_loader(cfg, n_iter)

        model = RecurrenceManifold(d_embed=d_embed)
        if use_test:
            z_pred = model.fit_predict(X_test)
            maxcs.append(get_maxes(*comp_ccorr(z_test, z_pred))[1])
        else:
            y_recon = model.fit_predict(X_train)
            tau, c = comp_ccorr(z_train, y_recon)
            maxcs.append(get_maxes(tau, c)[1])

    save_results(fname=interim_res_path / 'shrec_res.csv', r=maxcs, N=N, method='ShRec', dataset=family_key)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_FAMILY_KEY_MAP = {
    'logmaps': 'logmaps',
    'tentmaps': 'tentmaps',
    'lorenz': 'lorenz',
}

_SKLEARN_MODELS = {
    'pca': ('sklearn.decomposition', 'PCA'),
    'ica': ('sklearn.decomposition', 'FastICA'),
    'kpca': ('sklearn.decomposition', 'KernelPCA'),
    'cca': ('sklearn.cross_decomposition', 'CCA'),
}


def _import_model(module_path: str, class_name: str):
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def main():
    parser = argparse.ArgumentParser(description='Run a baseline method on an experiment family.')
    parser.add_argument('--family', required=True, choices=sorted(_FAMILY_KEY_MAP))
    parser.add_argument('--method', required=True, choices=sorted(_SKLEARN_MODELS))
    parser.add_argument('--config', default=None)
    args = parser.parse_args()

    family_key = _FAMILY_KEY_MAP[args.family]
    method_name = args.method.upper()

    # Look up sklearn model
    if args.method not in _SKLEARN_MODELS:
        print(f'Error: method "{args.method}" not supported via sklearn runner. '
              f'Supported: {sorted(_SKLEARN_MODELS)}')
        raise SystemExit(1)

    mod_path, cls_name = _SKLEARN_MODELS[args.method]
    model_class = _import_model(mod_path, cls_name)

    def model_factory(method_cfg: dict):
        """Create model from method config, stripping known non-model keys."""
        kwargs = dict(method_cfg)
        # Remove keys that are family-level config, not sklearn params
        kwargs.pop('d_embed', None)
        if args.method == 'cca':
            kwargs.pop('max_iter', None)  # keep max_iter — it IS a sklearn param
        return model_class(**kwargs)

    run_baseline_method(family_key, method_name, model_factory, config_path=args.config)


if __name__ == '__main__':
    main()
