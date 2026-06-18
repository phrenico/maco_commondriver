"""
Unified canonical configuration for all experiment families, datagen, and plotting scripts.

- All path constants and realization counts are defined once here.
- All CONFIG_<FAMILY> dicts are migrated from scripts/experiments/config.py.
- Directory creation is performed at import time to preserve legacy behavior.
"""

from pathlib import Path
from typing import Final
import numpy as np

# === Canonical Paths ===
project_path: Final[Path] = Path(__file__).resolve().parents[1]
data_root: Final[Path] = project_path / 'data'
artifacts_root: Final[Path] = project_path / 'paper_artifacts'
results_root: Final[Path] = artifacts_root / 'results'
interim_results_root: Final[Path] = results_root / 'interim'
final_results_root: Final[Path] = results_root / 'final'
figures_root: Final[Path] = artifacts_root / 'figures'
misc_figure_path: Final[Path] = figures_root
lorenz_htune_figure_path: Final[Path] = figures_root / 'lorenz_htune'
lorenz_data_path: Final[Path] = data_root / 'lorenz'
lorenz_data_path_template: Final[str] = str(lorenz_data_path / 'lorenz_{}.npz')

# Per-family final result paths (used by plot scripts)
logmaps_final_res_path: Final[Path] = final_results_root
tentmaps_final_res_path: Final[Path] = final_results_root
noise_length_final_res_path: Final[Path] = final_results_root / 'noise_length'
example_logmap_final_res_path: Final[Path] = final_results_root / 'example_logmap'

# === Directory creation (preserve legacy behavior) ===
for path in (
    data_root,
    artifacts_root,
    results_root,
    interim_results_root,
    final_results_root,
    figures_root,
    misc_figure_path,
    lorenz_htune_figure_path,
):
    path.mkdir(parents=True, exist_ok=True)

# === Realization counts (canonical) ===
N = 50
example_logmap_realizations: Final[int] = 1
logmap_realizations: Final[int] = N
tentmap_realizations: Final[int] = N
lorenz_realizations: Final[int] = N
lorenz_htune_realizations: Final[int] = N
noise_length_realizations: Final[int] = 10

# === CONFIG_<FAMILY> dicts (migrated and unified) ===
CONFIG_LOGMAPS = {
    'datagen': {
        'seed': None,  # int or None; if None, realization index i is used as seed
        'N': logmap_realizations,
        'n': 6_000,
        'rint': (3.8, 4.0),
        'A0': np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=float),
        'A':  np.array([[1.0, 0.0, 0.0], [0.3, 1.0, 0.0], [0.4, 0.0, 1.0]], dtype=float),
    },
    'preprocessing': {
        'train_split': 1.0 / 3,
        'valid_split': 1.0 / 3,
        'd_embed': 3,
    },
    'paths': {
        'interim_res_path': str(interim_results_root / 'logmaps'),
        'final_res_path':   str(final_results_root),
        'figure_path':      str(figures_root),
    },
    'maco': {
        'n_epochs':   300,
        'n_models':   10,
        'batch_size': 1_000,
        'lr':         1e-2,
        'dx':         1,
        'dy':         2,
        'dz':         1,
        'n_hidden':   20,
        'tau':        1,
    },
    'methods': {
        'pca':    {'n_components': 1},
        'kpca':   {'n_components': 1, 'kernel': 'rbf'},
        'ica':    {'n_components': 2},
        'cca':    {'n_components': 1},
        'dcca':   {'features': [3, 3], 'layers': [20, 20, 1]},
        'dca':    {'n_components': 1, 'T': 5, 'n_init': 10},
        'sfa':    {'n_components': 1, 'poly_degree': 2},
        'shrec':  {'d_embed': 3},
        'anisom': {'d_embed': 3, 'd_grid': 2, 'sizes': [40, 20], 'epochs': 4},
        'random': {},
    },
}

CONFIG_TENTMAPS = {
    'datagen': {
        'seed': None,  # int or None; if None, realization index i is used as seed
        'N': tentmap_realizations,
        'n': 6_000,
        'aint': (2.0, 10.0),
        'A0': np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=float),
    },
    'preprocessing': {
        'train_split': 1.0 / 3,
        'valid_split': 1.0 / 3,
        'd_embed': 2,
    },
    'paths': {
        'interim_res_path': str(interim_results_root / 'tentmaps'),
        'final_res_path':   str(final_results_root),
        'figure_path':      str(figures_root / 'misc'),
    },
    'maco': {
        'n_epochs':   300,
        'n_models':   10,
        'batch_size': 1_000,
        'lr':         1e-2,
        'dx':         1,
        'dy':         2,
        'dz':         1,
        'n_hidden':   20,
        'tau':        1,
    },
    'methods': {
        'pca':    {'n_components': 1},
        'kpca':   {'n_components': 1, 'kernel': 'rbf'},
        'ica':    {'n_components': 5, 'd_embed': 3},
        'cca':    {'n_components': 1, 'd_embed': 3},
        'dcca':   {'features': [2, 2], 'layers': [20, 20, 1]},
        'dca':    {'n_components': 5, 'T': 5, 'n_init': 10, 'd_embed': 3},
        'sfa':    {'n_components': 1, 'poly_degree': 2, 'd_embed': 3},
        'shrec':  {'d_embed': 3},
        'anisom': {'d_embed': 3, 'd_grid': 2, 'sizes': [40, 20], 'epochs': 1},
        'random': {'d_embed': 3},
    },
}

CONFIG_LORENZ = {
    'data': {
        'seed': None,  # int or None; if None, realization index i is used as seed
        'N': lorenz_realizations,
        'data_path_template': lorenz_data_path_template,
    },
    'preprocessing': {
        'train_split': 1.0 / 3,
        'valid_split': 1.0 / 3,
    },
    'paths': {
        'interim_res_path': str(interim_results_root / 'lorenz'),
        'final_res_path':   str(final_results_root),
        'figure_path':      str(figures_root),
    },
    'maco': {
        'n_epochs':   200,
        'n_models':   10,
        'batch_size': 1_000,
        'lr':         1e-2,
        'dx':         3,
        'dy':         3,
        'dz':         1,
        'n_hidden':   20,
        'tau':        1,
    },
    'methods': {
        'pca':    {'n_components': 5},
        'kpca':   {'n_components': 5, 'kernel': 'rbf'},
        'ica':    {'n_components': 5},
        'cca':    {'n_components': 1, 'max_iter': 500},
        'dcca':   {'d_embed': 3, 'features': [3, 3], 'layers': [20, 20, 1]},
        'dca':    {'n_components': 5, 'T': 5, 'n_init': 10},
        'sfa':    {'n_components': 3, 'poly_degree': None},
        'shrec':  {'d_embed': 3},
        'random': {},
    },
}

CONFIG_EXAMPLE_LOGMAP = {
    'datagen': {
        'seed': None,  # int or None; if None, realization index i is used as seed
        'N': example_logmap_realizations,
        'n': 10_000,
        'rint': (3.8, 4.0),
        'A0': np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=float),
        'A':  np.array([[1.0, 0.0, 0.0], [0.3, 1.0, 0.0], [0.4, 0.0, 1.0]], dtype=float),
    },
    'preprocessing': {
        'trainset_size': 80,
        'testset_size':  10,
        'validset_size': 10,
    },
    'paths': {
        'final_res_path': str(final_results_root / 'example_logmap'),
        'figure_path': str(figures_root),
    },
    'maco': {
        'n_epochs':   2_000,
        'n_models':   10,
        'batch_size': 1_000,
        'lr':         1e-2,
        'dx':         1,
        'dy':         2,
        'dz':         1,
        'n_hidden':   20,
        'tau':        1,
        'device':     'cpu',
    },
}

CONFIG_NOISE_LENGTH = {
    'datagen': {
        'seed': None,  # int or None; if None, realization index i is used as seed
        'nvars': 3,
        'N':     noise_length_realizations,
        'rint':  (3.8, 4.0),
        'A0': np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=float),
    },
    'length_sweep': {
        'Ls': list(range(100, 1_000, 200)) + list(range(1_000, 3_001, 1_000)),
        'n':  3_000,
    },
    'noise_sweep': {
        'Ls': list(10.0 ** np.arange(-3, 0.5, 0.25)),
        'n':  1_000,
    },
    'preprocessing': {
        'trainset_size': 80,
        'testset_size':  10,
        'validset_size': 10,
    },
    'paths': {
        'final_res_path': str(final_results_root / 'noise_length'),
        'figure_path':    str(figures_root),
    },
    'maco': {
        'n_epochs':   100,
        'n_models':   10,
        'batch_size': 500,
        'lr':         1e-2,
        'dx':         1,
        'dy':         2,
        'dz':         1,
        'n_hidden':   20,
        'tau':        1,
    },
}

CONFIG_LORENZ_HTUNE = {
    'sweep': {
        'ns_components': list(range(1, 7)),
    },
    'preprocessing': {
        'train_split': 0.5,
        'valid_split': 0.25,
    },
    'paths': {
        'interim_res_path': str(interim_results_root / 'lorenz_htune'),
        'final_res_path':   str(final_results_root / 'lorenz_htune'),
        'figure_path':      str(figures_root),
    },
    'data': {
        'seed': None,  # int or None; if None, realization index i is used as seed
        'N': 25,
        'data_path_template': lorenz_data_path_template,
    },
}

CONFIG_COMPARISON_PLOTS = {
    'paths': {
        'logmaps_final_res_path': str(final_results_root),
        'tentmaps_final_res_path': str(final_results_root),
        'lorenz_final_res_path': str(final_results_root),
        'figure_path': str(figures_root),
    },
}
