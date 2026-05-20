# sfa_env

`sfa_env` is the UV project used for registry steps mapped to the Slow Feature Analysis backend.

Typical direct commands:

```bash
cd envs/sfa_env
uv run python -m scripts.experiments.logmaps.gen_sfa_res --help
uv run python -m scripts.experiments.lorenz_hypertune.genres_sfa_htune --help
```

Use the family runner from the repository root when you want the registry-controlled execution order.
