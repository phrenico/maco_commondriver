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
    ├── maco_env/          # MaCo environment (PyTorch, etc.)
    ├── dca_env/           # DCA environment
    ├── dcca_env/          # DCCA environment
    ├── shrec_env/         # SHREC environment
    └── sfa_env/           # SFA environment
├── scripts/             # Experiment runners, data generation, and figure-generation scripts for the article
│   ├── config.py          # basic configuration parameters
│   ├── datagen_scripts/   # explicit data-generation entry points
│   ├── plots/             # figure-generation scripts driven by final CSVs
│   └── experiments/       # experiment and result-generation workflows
├── paper_artifacts/     # curated paper figures and final result tables (tracked)
├── data/                # placeholder for generated datasets (e.g. Lorenz trajectories)
└── tests/               # smoke and regression tests
```

## Installation

From the repository root:

```bash
python -m pip install -e .
```

This installs the core package and the MaCo-based workflows.

## Reproducible Experiment Environments

Experiment workflows use pre-configured uv project environments in `envs/`:

- `envs/maco_env` — MaCo (PyTorch-based)
- `envs/dca_env` — Dynamical Component Analysis
- `envs/dcca_env` — Deep Canonical Correlation Analysis
- `envs/shrec_env` — ShRec method
- `envs/sfa_env` — Slow Feature Analysis

Each environment has a `pyproject.toml` with its dependencies and an editable reference to the repository root. **No additional setup is needed**; the toml files are pre-configured and ready to use.

The family runner enforces strict UV-only execution: if `envs/<env_name>/pyproject.toml` is missing for a registry step, it fails immediately with a clear error.

## Data Generation

The logistic-map and tent-map workflows generate their synthetic datasets inside the experiment scripts.

The Lorenz comparison workflow reads `data/lorenz/lorenz_*.npz`. To regenerate those files:

```bash
python scripts/datagen_scripts/lorenz_datgen.py
```

## Main Comparison Workflows

### Run All Experiments

To run all 7 registered families and generate comparison plots in one go:

```bash
python -m scripts.run_all
```

This orchestrates:
1. `logmaps` — Logistic map comparison suite
2. `tentmaps` — Tent map comparison suite
3. `lorenz` — Lorenz system comparison suite
4. `lorenz_htune` — Lorenz hyperparameter tuning
5. `example_logmap` — Worked logistic-map MaCo example
6. `noise_length` — MaCo noise and length dependence analysis
7. `dummy_experiment` — Test family for quick validation

Each family runs its methods through its mapped UV environment, then comparison plots are generated and written to `paper_artifacts/figures/`.

For a dry-run preview:

```bash
python -m scripts.run_all --dry-run
```

### Run Individual Families

To run a single family:

```bash
cdriver-run-family <family>
```

or equivalently:

```bash
python -m scripts.experiments.run_family <family>
```

Each step is executed through `uv run` inside its mapped `envs/<env_name>/` project directory.

Examples:

```bash
cdriver-run-family logmaps
cdriver-run-family tentmaps
cdriver-run-family lorenz
cdriver-run-family lorenz_htune
cdriver-run-family example_logmap
cdriver-run-family noise_length
```

### Method Inventory

- **Logmaps**: PCA, kPCA, ICA, CCA, DCA, DCCA, SFA, ShRec, AniSOM, random control, MaCo
- **Tentmaps**: PCA, kPCA, ICA, CCA, DCA, DCCA, SFA, ShRec, AniSOM, random control, MaCo
- **Lorenz**: PCA, ICA, CCA, DCA, DCCA, SFA, ShRec, random control, MaCo
- **Lorenz Hypertune**: Hyperparameter tuning for the Lorenz dataset (ICA, PCA, DCA, SFA)
- **Noise / Length Analysis**: MaCo robustness to noise and data length
- **Example Logmap**: Step-by-step MaCo walkthrough

For compatibility, legacy bash wrappers also exist but are not maintained:

- `bash scripts/experiments/logmaps/Z_run_all.sh`
- `bash scripts/experiments/tentmaps/Z_run_all.sh`
- `bash scripts/experiments/lorenz/Z_run_all.sh`

## Additional Module-Level Workflows

Beyond the main family runner, individual modules can be invoked directly for ad-hoc exploration:

```bash
# Noise/length analysis (not part of main families)
python -m scripts.experiments.noise_length.maco_noise
python -m scripts.experiments.noise_length.maco_length

# Lorenz hyperparameter tuning
python -m scripts.experiments.lorenz.lorenz_hypertune.run_hypertune
```

These bypass the family orchestrator and run directly in their respective UV environments via their module paths.

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

## Notes

- `scripts/config.py` resolves the repository root dynamically; commands should be run from a normal checkout without editing machine-specific paths.
- `data/`, `results/`, and `figures/` are reproducible output locations (gitignored); pre-computed paper outputs are in `paper_artifacts/`.
- The maintained execution path is centered on `cdriver-run-family` (console script), backed by `scripts/experiments/run_family.py` and `scripts/experiments/experiment_registry.py`.