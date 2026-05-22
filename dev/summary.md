# Commit-Style Summary

Make comparison plotting fully config-driven

- require explicit `CONFIG_COMPARISON_PLOTS` for the shared comparison plot
- fail fast in `plot_comparisons` and `run_all` when `--config` is missing
- validate required comparison-plot path keys up front
- add a dedicated comparison-plot config template
- add explicit `CONFIG_COMPARISON_PLOTS` support in `config_runall.py`
- extend execution-path tests for shared comparison plot success and failure cases
- update docs for the mandatory-config comparison plot workflow
