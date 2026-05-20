# dca_env

`dca_env` is the UV project used for registry steps mapped to the Dynamical Components Analysis backend.

Typical direct commands:

```bash
cd envs/dca_env
uv run python -m scripts.experiments.logmaps.gen_dca_res --help
uv run python -m scripts.experiments.lorenz_hypertune.genres_dca_htune --help
```

For the maintained end-to-end workflow, prefer running the family orchestrator from the repository root.
