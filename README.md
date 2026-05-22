# Common Driver Reconstruction

Code, experiment runners, and figure-generation workflows for the paper _Reconstructing shared dynamics with a deep neural network_.

## Citation

If you use this repository, please cite the paper:

> Zsigmond Benko and Zoltan Somogyvari. _Reconstructing shared dynamics with a deep neural network_. arXiv:2105.02322 [cs.NE], 2021. https://doi.org/10.48550/arXiv.2105.02322

and the ASOM paper:
> Zsigmond Benkő, Marcell Stippinger, Attila Bencze, Fülöp Bazsó, András Telcs, and Zoltán Somogyvári. _Inference of hidden common driver dynamics by anisotropic self-organizing neural networks_. _Neural Networks_, 194, 2026, 108113. ISSN 0893-6080. https://doi.org/10.1016/j.neunet.2025.108113


[![Test](https://github.com/benkozsigmond/maco_commondriver/actions/workflows/test.yml/badge.svg)](https://github.com/benkozsigmond/maco_commondriver/actions/workflows/test.yml)

---

## Repository Layout

```text
.
├── cdriver/             # Package (reusable code and core implementations)
│   ├── datagen/           # synthetic dynamical-system generators
│   ├── evaluate/          # reconstruction metrics and evaluation helpers
│   ├── network/           # model implementations (MaCo, AniSOM)
│   │   ├── maco.py          # Mapper-Coach implementation
│   │   └── anisom.py        # AniSOM implementation
│   ├── preprocessing/     # time-delay embedding and dataset splitting
│   ├── savers/            # result-table writing helpers
│   └── visuals/           # visualization helpers
├── data/                # placeholder for generated datasets (gitignored; .gitkeep tracked)
├── dev/                 # audit reports, fix plans, and session summaries
├── paper_artifacts/     # curated paper figures and final result tables (tracked)
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
│   └── experiments/       # experiment runners and method_runner.py
├── tests/               # unit, integration, and smoke tests (64 tests, 12 files)
└── .github/workflows/   # CI pipeline (pytest on push/PR)
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
- `envs/dca_env` — Dynamical Component Analysis (git dep pinned)
- `envs/dcca_env` — Deep Canonical Correlation Analysis (via mvlearn)
- `envs/shrec_env` — ShRec method (git dep pinned)
- `envs/sfa_env` — Slow Feature Analysis (via sklearn-sfa)

Each environment has a `pyproject.toml` with its dependencies and an editable reference to the
repository root. **No additional setup is needed**; the toml files are pre-configured. `uv` will
create and populate the environments on first run automatically.

The family runner enforces strict uv-only execution: every step runs via `uv run` inside its
environment directory.

All git-sourced dependencies (`dynamicalcomponentsanalysis`, `shrec`) are pinned to a stable
revision to prevent silent breakage from upstream changes.

---

## Data Generation

All synthetic data generators support an optional `seed` field in their config dicts.
When `seed` is `None` (the default), the realization index `i` is used as seed — preserving
backward compatibility with paper results. Setting an explicit `seed` gives reproducible subsets.

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

### Run Individual Methods

For sklearn-baseline methods (PCA, ICA, CCA, KPCA), a parameterized runner is available:

```bash
python -m scripts.experiments.method_runner --family lorenz --method pca [--config path/to/config.py]
```

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
- `scripts/config_templates/config_comparison_plots_template.py`

Step-by-step usage tutorial:

- `scripts/config_templates/config_tutorial.md`

Relative output paths inside a config dict are resolved from the repository root; result
directories are created automatically when CSV outputs are written.

For family-owned plots, define `paths.figure_path` in the same config dict that controls
the experiment outputs. This is used by `example_logmap`, `noise_length`, `lorenz_htune`,
and the standalone plot entrypoints for `logmaps`, `tentmaps`, and `lorenz`.

For the shared aggregate comparison figure, define a separate `CONFIG_COMPARISON_PLOTS`
block and pass it explicitly with `--config`. The shared comparison plot does not run
without an explicit config file.

For standalone template configs, matrix-valued datagen fields that are passed into NumPy
operations are declared with `np.array(...)` rather than plain Python lists.

### Method Inventory

- **Logmaps**: PCA, kPCA, ICA, CCA, DCA, DCCA, SFA, ShRec, AniSOM, random control, MaCo
- **Tentmaps**: PCA, kPCA, ICA, CCA, DCA, DCCA, SFA, ShRec, AniSOM, random control, MaCo
- **Lorenz**: PCA, ICA, CCA, DCA, DCCA, SFA, ShRec, random control, MaCo
- **Lorenz Hypertune**: Hyperparameter tuning for the Lorenz dataset (ICA, PCA, DCA, SFA)
- **Noise / Length Analysis**: MaCo robustness to noise and data length
- **Example Logmap**: Step-by-step MaCo walkthrough

## Figure Generation

The shared comparison plot is generated as the final step of `python -m scripts.run_all`
only when you provide an explicit `--config` file that defines `CONFIG_COMPARISON_PLOTS`.

Manual figure generation:

```bash
# Shared aggregate comparison plot (requires explicit CONFIG_COMPARISON_PLOTS)
python -m scripts.plots.plot_comparisons --config path/to/config.py

# Noise and length analysis plots
python -m scripts.plots.noise_length.plot_noise_length --config path/to/config.py

# Family-owned standalone plot entrypoints
python -m scripts.plots.logmaps.comparison_plot_logmap --config path/to/config.py
python -m scripts.plots.tentmaps.comparison_plot_tentmap --config path/to/config.py
python -m scripts.plots.lorenz.comparison_plot_lorenz --config path/to/config.py
```

Dataset-specific plot scripts:

- `scripts/plots/logmaps/` — Logistic map plots
- `scripts/plots/tentmaps/` — Tent map plots
- `scripts/plots/lorenz/` — Lorenz system plots
- `scripts/plots/example_logmap/` — Example walkthrough plots
- `scripts/plots/noise_length/` — Robustness analysis plots

## Testing

The test suite has 64 tests across 12 files covering:
- Core algorithms: MaCo forward pass, training, loss computation
- Data generators: LogMap, TentMap, Lorenz ODE (seeded reproducibility, boundary conditions, chaos)
- Evaluation metrics: cross-correlation, max-lag detection, linear regression
- Preprocessing: time-delay embedding, data splitting, edge cases
- Infrastructure: experiment registry, execution paths, config loading, result combining
- Integration: end-to-end smoke tests (one realization, minimal epochs)

Run the test suite from the repository root:

```bash
pytest -q
```

If your user site has incompatible third-party pytest plugins installed, isolate the repo tests with:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/
```

CI runs on every push and PR via GitHub Actions (`.github/workflows/test.yml`) across Python 3.10 and 3.12.

## Notes

`scripts/config_runall.py` remains the canonical built-in config module for the experiment families. For the shared comparison plot, it is only used when you pass it explicitly via `--config`; it is not an implicit backup source.

### Configuration

As of May 2026, the configuration system is unified under `scripts/config_runall.py`. All
experiment, datagen, and plot scripts import from it. External config overrides (`--config`)
still require `CONFIG_<FAMILY>` dicts. See `scripts/config_templates/config_tutorial.md`
for step-by-step guidance.

The shared aggregate comparison figure generated by `scripts.plots.plot_comparisons` is still
outside the family-config path system for now; it continues to read the canonical default output
locations.

### Reproducibility

`data/`, `paper_artifacts/results/`, and `paper_artifacts/figures/` are reproducible output
locations (gitignored). Pre-computed paper outputs are tracked in `paper_artifacts/`.