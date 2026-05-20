# maco_env

`maco_env` is the default UV project for registry steps mapped to `maco_env`.

It covers:
- MaCo runs
- AniSOM runs
- several baseline sklearn-based comparison methods
- combine steps for the main experiment families
- the noise/length workflows

Most users should invoke these steps from the repository root through `python -m scripts.experiments.run_family <family>`.

For direct module execution:

```bash
cd envs/maco_env
uv run python -m scripts.experiments.logmaps.gen_maco_res --help
uv run python -m scripts.experiments.noise_length.maco_noise --help
```

The workspace package `cdriver` is linked in editable mode through the UV workspace.
