# Common Driver Reconstruction

Code, experiment runners, and figure-generation workflows for the paper _Reconstructing shared dynamics with a deep neural network_.

## Citation

If you use this repository, please cite the paper:

> Zsigmond Benko and Zoltan Somogyvari. _Reconstructing shared dynamics with a deep neural network_. arXiv:2105.02322 [cs.NE], 2021. https://doi.org/10.48550/arXiv.2105.02322

and the ASOM paper:
> Zsigmond Benkő, Marcell Stippinger, Attila Bencze, Fülöp Bazsó, András Telcs, and Zoltán Somogyvári. _Inference of hidden common driver dynamics by anisotropic self-organizing neural networks_. _Neural Networks_, 194, 2026, 108113. ISSN 0893-6080. https://doi.org/10.1016/j.neunet.2025.108113


## Repository Layout

```text
.
├── cdriver/             # Package (reusable code and core implementations)
│   ├── datagen/           # synthetic dynamical-system generators
│   ├── evaluate/          # reconstruction metrics and evaluation helpers
│   ├── network/           # model implementations
│   │   ├── maco.py          # Mapper-Coach implementation
│   │   └── anisom.py        # AniSOM implementation
│   ├── preprocessing/     # time-delay embedding and dataset splitting
│   ├── savers/            # result-table writing helpers
│   └── visuals/           # visualization helpers
├── envs/                # uv project directories for reproducible experiment environments
│   ├── maco_env/          # MaCo environment (PyTorch, etc.)
│   ├── dca_env/           # DCA environment
│   ├── dcca_env/          # DCCA environment
│   ├── shrec_env/         # SHREC environment
│   └── sfa_env/           # SFA environment
├── scripts/             # Experiment runners, data generation, and figure-generation scripts for the article
│   ├── config_runall.py   # canonical configuration for all experiments, datagen, and plots
│   ├── config_templates/ # standalone external-config templates and tutorial
│   ├── datagen_scripts/   # explicit data-generation entry points
│   ├── plots/             # figure-generation scripts driven by final CSVs
│   └── experiments/       # experiment and result-generation workflows
├── paper_artifacts/     # curated paper figures and final result tables (tracked)
├── data/                # placeholder for generated datasets (e.g. Lorenz trajectories)
└── tests/               # smoke and regression tests
```

## Installation

The core package is installed as an editable dependency inside each experiment environment.
No separate top-level install step is needed — the `uv` project files in `envs/` handle everything.

If you need the package available in your active Python environment (e.g. for running tests or
importing `cdriver` directly), install it manually:

```bash
pip install -e .
```

## Reproducible Experiment Environments

Experiment workflows use pre-configured uv project environments in `envs/`:

- `envs/maco_env` — MaCo (PyTorch-based)
- `envs/dca_env` — Dynamical Component Analysis
- `envs/dcca_env` — Deep Canonical Correlation Analysis
- `envs/shrec_env` — ShRec method
- `envs/sfa_env` — Slow Feature Analysis

Each environment has a `pyproject.toml` with its dependencies and an editable reference to the
repository root. **No additional setup is needed**; the toml files are pre-configured. `uv` will
create and populate the environments on first run automatically.

The family runner enforces strict uv-only execution: if `envs/<env_name>/pyproject.toml` is missing
for a registry step, it fails immediately with a clear error.

## Data Generation

The logistic-map and tent-map workflows generate their synthetic datasets inside the experiment scripts.

The Lorenz comparison workflow reads `data/lorenz/lorenz_*.npz`. To regenerate those files:

```bash
python scripts/datagen_scripts/lorenz_datgen.py
```

## Main Comparison Workflows

### Run All Experiments

To run all registered families and generate comparison plots in one go:

```bash
python -m scripts.run_all
```

This orchestrates (in order):
1. `logmaps` — Logistic map comparison suite
2. `tentmaps` — Tent map comparison suite
3. `lorenz` — Lorenz system comparison suite
4. `lorenz_htune` — Lorenz hyperparameter tuning (ICA, PCA, DCA, SFA)
5. `example_logmap` — Worked logistic-map MaCo example
6. `noise_length` — MaCo noise and length dependence analysis

Each family runs its methods through its mapped UV environment, then comparison plots are generated and written to `paper_artifacts/figures/`.

For a dry-run preview (prints all subprocess commands without executing them):

```bash
python -m scripts.run_all --dry-run
```

To forward a custom config to every family:

```bash
python -m scripts.run_all --dry-run --config path/to/custom_config.py
```

### Run Individual Families

```bash
cdriver-run-family <family>
```

Available families: `logmaps`, `tentmaps`, `lorenz`, `lorenz_htune`, `example_logmap`,
`noise_length`.

Each step runs via `uv run` inside its mapped `envs/<env_name>/` directory.

To override config parameters, create a Python file defining the matching `CONFIG_<FAMILY>` dict
(e.g. `CONFIG_LOGMAPS = {...}`, `CONFIG_LORENZ = {...}`) and pass it via `--config`. If the file is missing or the dict is
absent, the runner **fails immediately** — there is no silent fallback:

```bash
cdriver-run-family logmaps --config path/to/custom_config.py
cdriver-run-family lorenz_htune --config path/to/custom_config.py
```

### Standalone Config Templates

Ready-to-copy standalone templates are available in `scripts/config_templates/`.
Each file is self-contained and defines one `CONFIG_<FAMILY>` dict directly (no import from `scripts/config_runall.py` required).

Available templates:

- `scripts/config_templates/config_logmaps_template.py`
- `scripts/config_templates/config_tentmaps_template.py`
- `scripts/config_templates/config_lorenz_template.py`
- `scripts/config_templates/config_lorenz_htune_template.py`
- `scripts/config_templates/config_example_logmap_template.py`
- `scripts/config_templates/config_noise_length_template.py`

Step-by-step usage tutorial:

- `scripts/config_templates/config_tutorial.md`

Relative output paths inside a config dict are resolved from the repository root; result
directories are created automatically when CSV outputs are written.

### Method Inventory

- **Logmaps**: PCA, kPCA, ICA, CCA, DCA, DCCA, SFA, ShRec, AniSOM, random control, MaCo
- **Tentmaps**: PCA, kPCA, ICA, CCA, DCA, DCCA, SFA, ShRec, AniSOM, random control, MaCo
- **Lorenz**: PCA, ICA, CCA, DCA, DCCA, SFA, ShRec, random control, MaCo
- **Lorenz Hypertune**: Hyperparameter tuning for the Lorenz dataset (ICA, PCA, DCA, SFA)
- **Noise / Length Analysis**: MaCo robustness to noise and data length
- **Example Logmap**: Step-by-step MaCo walkthrough

## Figure Generation

Comparison plots are generated automatically as the final step of `python -m scripts.run_all`.

Manual figure generation:

```bash
# Comparison plots (requires final CSVs in paper_artifacts/results/)
python -m scripts.plots.plot_comparisons

# Noise and length analysis plots
python -m scripts.plots.noise_length.plot_noise_length
```

Dataset-specific plot scripts:

- `scripts/plots/logmaps/` — Logistic map plots
- `scripts/plots/tentmaps/` — Tent map plots
- `scripts/plots/lorenz/` — Lorenz system plots
- `scripts/plots/example_logmap/` — Example walkthrough plots
- `scripts/plots/noise_length/` — Robustness analysis plots

## Testing

Run the lightweight test suite from the repository root:

```bash
pytest -q
```

If your user site has incompatible third-party pytest plugins installed, isolate the repo tests with:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
```

## Notes

`scripts/config_runall.py` is the canonical default config module used by all experiment runners, datagen, and plotting scripts.
# Migration Note

As of May 2026, the configuration system has been unified:
- All configuration is now in `scripts/config_runall.py`.
- Legacy files `scripts/config.py` and `scripts/experiments/config.py` have been removed.
- All experiment, datagen, and plot scripts import from `scripts/config_runall.py`.
- External config overrides (`--config`) must still define `CONFIG_<FAMILY>` dicts as before.
- `data/`, `results/`, and `figures/` are reproducible output locations (gitignored); pre-computed paper outputs are in `paper_artifacts/`.
- The maintained execution path is centered on `cdriver-run-family` (console script), backed by `scripts/experiments/run_family.py` and `scripts/experiments/experiment_registry.py`.