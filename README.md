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
├── scripts/             # Result-generation scripts for the article
│   ├── config.py          # basic configuration parameters
│   ├── datagen_scripts/   # explicit data-generation entry points
│   ├── figgen/            # figure-generation scripts driven by final CSVs
│   └── resgen/            # experiment and result-generation workflows
├── data/                # placeholder for generated datasets (e.g. Lorenz trajectories)
├── results/             # generated intermediate and final result tables
├── figures/             # generated figures
└── tests/               # smoke and regression tests
```

## Installation

From the repository root:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

This installs the core package and the MaCo-based workflows. Some comparison baselines still run in separate conda environments, and those environment names are encoded in `scripts/resgen/experiment_registry.py`.

## Data Generation

The logistic-map and tent-map workflows generate their synthetic datasets inside the experiment scripts.

The Lorenz comparison workflow reads `data/lorenz/lorenz_*.npz`. To regenerate those files:

```bash
python scripts/datagen_scripts/lorenz_datgen.py
```

## Main Comparison Workflows

The canonical family runner is:

```bash
python -m scripts.resgen.run_family <family>
```

Supported families:

- `logmaps`
- `tentmaps`
- `lorenzs`

Examples:

```bash
python -m scripts.resgen.run_family logmaps
python -m scripts.resgen.run_family tentmaps
python -m scripts.resgen.run_family lorenzs
```

For compatibility, the family shell wrappers still exist:

- `bash scripts/resgen/logmaps/Z_run_all.sh`
- `bash scripts/resgen/tentmaps/Z_run_all.sh`
- `bash scripts/resgen/lorenzs/Z_run_all.sh`

Current method inventory:

- Logmaps: PCA, kPCA, ICA, CCA, DCA, DCCA, SFA, ShRec, AniSOM, random control, MaCo
- Tentmaps: PCA, kPCA, ICA, CCA, DCA, DCCA, SFA, ShRec, AniSOM, random control, MaCo
- Lorenzs: PCA, ICA, CCA, DCA, DCCA, SFA, ShRec, random control, MaCo

## Additional Resgen Workflows

The `scripts/resgen/` tree currently contains these maintained sub-workflows:

- `example_logmap/`: worked logistic-map MaCo example
- `logmaps/`: full logistic-map comparison suite
- `tentmaps/`: full tent-map comparison suite
- `lorenzs/`: full Lorenz comparison suite
- `lorenzs/lorenz_hypertune/`: Lorenz hyperparameter tuning
- `noise_length/`: MaCo noise dependence and data-length dependence analyses

Representative commands:

```bash
python scripts/resgen/example_logmap/gen_exampleResults.py
python scripts/resgen/noise_length/maco_noise.py
python scripts/resgen/noise_length/maco_length.py
```

## Figure Generation

Figure scripts consume the final CSVs under `results/final/`.

Representative entry points:

```bash
python -m scripts.figgen.plot_comparisons
python scripts/figgen/noise_length/plot_noise_length.py
```

Dataset-specific plotting scripts also live under:

- `scripts/figgen/logmaps/`
- `scripts/figgen/tentmaps/`
- `scripts/figgen/lorenzs/`
- `scripts/figgen/example_logmap/`
- `scripts/figgen/noise_length/`

## Notes

- `scripts/config.py` resolves the repository root dynamically; commands should be run from a normal checkout without editing machine-specific paths.
- `data/`, `results/`, and `figures/` are reproducible output locations, not the core source tree.
- The maintained execution path is now centered on `scripts/resgen/run_family.py` and `scripts/resgen/experiment_registry.py`.

