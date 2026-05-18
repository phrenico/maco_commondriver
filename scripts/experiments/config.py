"""Central experiment configuration.

Each experiment family is represented by a ``CONFIG_<FAMILY>`` dict with the
following top-level keys (where applicable):

  datagen        — data-generation parameters passed to the generator function
  data           — for families that read pre-generated files (lorenz)
  preprocessing  — train/valid/test splits, embedding dimensions
  paths          — output directories (strings relative to repo root;
                   call resolve_paths() to convert to absolute Path objects)
  maco           — MaCo training hyper-parameters
  methods        — method-specific parameters keyed by method name

To run with a custom config, create a Python file that defines one or more of
these dicts (e.g. ``CONFIG_LOGMAPS = {...}``) and pass it via ``--config``::

    python -m scripts.experiments.run_family logmaps --config my_config.py

Only the families you define are overridden; others fall back to this file.
"""

import numpy as np

# ============================================================================
# LOGISTIC MAPS
# ============================================================================

CONFIG_LOGMAPS = {
    'datagen': {
        'N': 50,                    # number of realizations
        'n': 6_000,                 # time-series length (3 × 2 000)
        'rint': (3.8, 4.0),         # r-parameter sampling interval
        'A0': np.array([[0, 0, 0],
                        [1, 0, 0],
                        [1, 0, 0]], dtype=float),
        'A':  np.array([[1.0, 0.0, 0.0],
                        [0.3, 1.0, 0.0],
                        [0.4, 0.0, 1.0]], dtype=float),
    },
    'preprocessing': {
        'train_split': 1.0 / 3,
        'valid_split': 1.0 / 3,
        'd_embed': 3,
    },
    'paths': {
        'interim_res_path': 'paper_artifacts/results/interim/logmaps',
        'final_res_path':   'paper_artifacts/results/final',
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
        'ica':    {'n_components': 2},  # fit 2 comps, take best correlated with z
        'cca':    {'n_components': 1},
        'dcca':   {'features': [3, 3], 'layers': [20, 20, 1]},
        'dca':    {'n_components': 1, 'T': 5, 'n_init': 10},
        'sfa':    {'n_components': 1, 'poly_degree': 2},
        'shrec':  {'d_embed': 3},
        'anisom': {'d_embed': 3, 'd_grid': 2, 'sizes': [40, 20], 'epochs': 4},
        'random': {},
    },
}

# ============================================================================
# TENT MAPS
# ============================================================================

CONFIG_TENTMAPS = {
    'datagen': {
        'N': 50,
        'n': 6_000,
        'aint': (2.0, 10.0),        # alpha-parameter sampling interval
        'A0': np.array([[0, 0, 0],
                        [1, 0, 0],
                        [1, 0, 0]], dtype=float),
    },
    'preprocessing': {
        'train_split': 1.0 / 3,
        'valid_split': 1.0 / 3,
        # Default d_embed for PCA / KPCA / DCCA / MaCo; individual methods below
        # override this via their own 'd_embed' key.
        'd_embed': 2,
    },
    'paths': {
        'interim_res_path': 'paper_artifacts/results/interim/tentmaps',
        'final_res_path':   'paper_artifacts/results/final',
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
        # ICA / CCA / DCA / SFA / random use d_embed=3 to match original scripts
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

# ============================================================================
# LORENZ SYSTEMS
# ============================================================================
# Lorenz experiments read pre-generated .npz files — no inline data generation.

CONFIG_LORENZ = {
    'data': {
        'N': 50,
        # Path template relative to repo root; use .format(n_iter) to get per-file path.
        'data_path_template': 'data/lorenz/lorenz_{}.npz',
    },
    'preprocessing': {
        'train_split': 1.0 / 3,
        'valid_split': 1.0 / 3,
    },
    'paths': {
        'interim_res_path': 'paper_artifacts/results/interim/lorenz',
        'final_res_path':   'paper_artifacts/results/final',
    },
    'maco': {
        'n_epochs':   200,          # NOTE: fewer epochs than logmaps
        'n_models':   10,
        'batch_size': 1_000,
        'lr':         1e-2,
        'dx':         3,            # NOTE: 3D input (lorenz has 3 observed vars)
        'dy':         3,
        'dz':         1,
        'n_hidden':   20,
        'tau':        1,
    },
    'methods': {
        'pca':    {'n_components': 5},  # NOTE: 5, not 1 — lorenz is higher-dim
        'ica':    {'n_components': 5},
        'cca':    {'n_components': 1, 'max_iter': 500},
        'dcca':   {'d_embed': 3, 'features': [3, 3], 'layers': [20, 20, 1]},
        'dca':    {'n_components': 5, 'T': 5, 'n_init': 10},
        'sfa':    {'n_components': 3, 'poly_degree': None},  # no poly expansion for lorenz
        'shrec':  {'d_embed': 3},
        'random': {},
    },
}

# ============================================================================
# EXAMPLE LOGISTIC MAP (single realization, long training)
# ============================================================================

CONFIG_EXAMPLE_LOGMAP = {
    'datagen': {
        'N': 1,
        'n': 10_000,
        'rint': (3.8, 4.0),
        'A0': np.array([[0, 0, 0],
                        [1, 0, 0],
                        [1, 0, 0]], dtype=float),
        'A':  np.array([[1.0, 0.0, 0.0],
                        [0.3, 1.0, 0.0],
                        [0.4, 0.0, 1.0]], dtype=float),
    },
    'preprocessing': {
        # Absolute sizes (not fractions) matching build_series_loaders interface
        'trainset_size': 80,
        'testset_size':  10,
        'validset_size': 10,
    },
    'paths': {
        'final_res_path': 'paper_artifacts/results/final/example_logmap',
    },
    'maco': {
        'n_epochs':   2_000,        # long training for the example
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

# ============================================================================
# NOISE & LENGTH SENSITIVITY
# ============================================================================

CONFIG_NOISE_LENGTH = {
    'datagen': {
        'nvars': 3,
        'N':     10,
        'rint':  (3.8, 4.0),
        'A0': np.array([[0, 0, 0],
                        [1, 0, 0],
                        [1, 0, 0]], dtype=float),
    },
    'length_sweep': {
        'Ls': list(range(100, 1_000, 200)) + list(range(1_000, 3_001, 1_000)),
        'n':  3_000,                # max(Ls)
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
        'final_res_path': 'paper_artifacts/results/final/noise_length',
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

# ============================================================================
# LORENZ HYPERPARAMETER TUNING
# ============================================================================
# Utility functions (compute4all, create_htune_df, plot_htune) stay in
# scripts/experiments/lorenz_hypertune/htune_config.py.

CONFIG_LORENZ_HTUNE = {
    'sweep': {
        'ns_components': list(range(1, 7)),
    },
    'preprocessing': {
        'train_split': 0.5,
        'valid_split': 0.25,
    },
    'paths': {
        'interim_res_path': 'paper_artifacts/results/interim/lorenz_htune',
        'final_res_path':   'paper_artifacts/results/final/lorenz_htune',
    },
    # Lorenz data paths reused from CONFIG_LORENZ
    'data': CONFIG_LORENZ['data'],
}
