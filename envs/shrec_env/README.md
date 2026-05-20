# shrec_env

`shrec_env` is the UV project used for registry steps mapped to the ShRec backend.

Typical direct commands:

```bash
cd envs/shrec_env
uv run python -m scripts.experiments.logmaps.gen_shrec_res --help
uv run python -m scripts.experiments.lorenz.gen_shrec_res --help
```

For full family runs, prefer the repository-root runner so the method ordering and combine step stay aligned with the registry.
