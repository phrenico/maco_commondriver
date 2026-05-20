# dcca_env

`dcca_env` is the UV project used for registry steps mapped to the DCCA backend.

Typical direct commands:

```bash
cd envs/dcca_env
uv run python -m scripts.experiments.logmaps.gen_dcca_res --help
uv run python -m scripts.experiments.lorenz.gen_dcca_res --help
```

For normal experiment runs, prefer `python -m scripts.experiments.run_family <family>` from the repository root.
