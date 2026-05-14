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

This installs the core package and the MaCo-based workflows. Some comparison baselines still run in separate conda environments, and those environment names are encoded in `scripts/experiments/experiment_registry.py`.

## Reproducible Environment Setup

Environment specs for the experiment runner are stored in `envs/` and mirror the env names used in `scripts/experiments/experiment_registry.py`:

- `envs/maco_rev1.yml`
- `envs/dca.yml`
- `envs/dcca_env.yml`
- `envs/shrec.yml`
- `envs/sfa.yml`

Create each environment:

```bash
conda env create -f envs/maco_rev1.yml
conda env create -f envs/dca.yml
conda env create -f envs/dcca_env.yml
conda env create -f envs/shrec.yml
conda env create -f envs/sfa.yml
```

Install this repository into each environment so `cdriver` and `scripts.*` modules are importable:

```bash
conda run -n maco_rev1 python -m pip install -e .
conda run -n dca python -m pip install -e .
conda run -n dcca_env python -m pip install -e .
conda run -n shrec python -m pip install -e .
conda run -n sfa python -m pip install -e .
```

Verify environment setup (strict: no skipping for missing envs):

```bash
pytest
```

The environment verification tests check that each required conda env exists and that method-specific imports succeed inside the corresponding env.

## Data Generation

The logistic-map and tent-map workflows generate their synthetic datasets inside the experiment scripts.

The Lorenz comparison workflow reads `data/lorenz/lorenz_*.npz`. To regenerate those files:

```bash
python scripts/datagen_scripts/lorenz_datgen.py
```

## Main Comparison Workflows

The canonical family runner is:

```bash
cdriver-run-family <family>
```

or equivalently:

```bash
python -m scripts.experiments.run_family <family>
```

Supported families:

- `logmaps`
- `tentmaps`
- `lorenz`

Examples:

```bash
cdriver-run-family logmaps
cdriver-run-family tentmaps
cdriver-run-family lorenz
```

For compatibility, the family shell wrappers still exist:

- `bash scripts/experiments/logmaps/Z_run_all.sh`
- `bash scripts/experiments/tentmaps/Z_run_all.sh`
- `bash scripts/experiments/lorenz/Z_run_all.sh`

Current method inventory:

- Logmaps: PCA, kPCA, ICA, CCA, DCA, DCCA, SFA, ShRec, AniSOM, random control, MaCo
- Tentmaps: PCA, kPCA, ICA, CCA, DCA, DCCA, SFA, ShRec, AniSOM, random control, MaCo
- Lorenz: PCA, ICA, CCA, DCA, DCCA, SFA, ShRec, random control, MaCo

## Additional Experiment Workflows

The `scripts/experiments/` tree currently contains these maintained sub-workflows:

- `example_logmap/`: worked logistic-map MaCo example
- `logmaps/`: full logistic-map comparison suite
- `tentmaps/`: full tent-map comparison suite
- `lorenz/`: full Lorenz comparison suite
- `lorenz/lorenz_hypertune/`: Lorenz hyperparameter tuning
- `noise_length/`: MaCo noise dependence and data-length dependence analyses

Representative commands:

```bash
python scripts/experiments/example_logmap/gen_exampleResults.py
python scripts/experiments/noise_length/maco_noise.py
python scripts/experiments/noise_length/maco_length.py
```

## Figure Generation

Figure scripts consume the final CSVs under `paper_artifacts/results/`.

Representative entry points:

```bash
python -m scripts.plots.plot_comparisons
python scripts/plots/noise_length/plot_noise_length.py
```

Dataset-specific plotting scripts also live under:

- `scripts/plots/logmaps/`
- `scripts/plots/tentmaps/`
- `scripts/plots/lorenz/`
- `scripts/plots/example_logmap/`
- `scripts/plots/noise_length/`

## Notes

- `scripts/config.py` resolves the repository root dynamically; commands should be run from a normal checkout without editing machine-specific paths.
- `data/`, `results/`, and `figures/` are reproducible output locations (gitignored); pre-computed paper outputs are in `paper_artifacts/`.
- The maintained execution path is centered on `cdriver-run-family` (console script), backed by `scripts/experiments/run_family.py` and `scripts/experiments/experiment_registry.py`.

