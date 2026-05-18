# --config Refactor Verification Summary

Date: 2026-05-18
Scope: Audit of `--config /path/to/file.py` refactor across `scripts/experiments` and orchestrators.

## PASS / FAIL Table

| Check | Result | Notes |
|---|---|---|
| 1. Central config infrastructure | PASS | `scripts/experiments/config_loader.py` exports `get_config`, `resolve_paths`. `scripts/experiments/config.py` defines all required `CONFIG_*` dicts. |
| 2. Per-script argparse/get_config/resolve_paths/main guard | FAIL | Target scripts broadly updated, but `scripts/experiments/lorenz/gen_maco_res.py` still has duplicate top-level execution code after the guarded block. |
| 3. `Z_combine_final_res.py` migration | FAIL | Main families use `cfg['paths']`, but `scripts/experiments/dummy_experiment/Z_combine_final_res.py` is still a stub and not config-driven. |
| 4. Old import elimination | FAIL | Legacy config imports remain in several experiment config/helper files (listed below). |
| 5. Path resolution correctness | FAIL | `_REPO_ROOT = Path(__file__).resolve().parents[3]` is generally correct and central config paths are relative strings, but Lorenz hypertune scripts still write via legacy `interim_save_path` import rather than `cfg['paths']`. |
| 6. Orchestrator propagation (`run_family`, `run_all`) | FAIL | `run_family` is correct. `run_all` includes `--config` propagation but is currently broken by indentation/control-flow issues (TabError, stray `else`). |
| 7. Dry-run smoke test (`run_family`) | PASS | Printed subprocess commands include `--config /tmp/fake_config.py` for `logmaps`, `lorenz`, and `noise_length`. |
| 8. Registry tests | PASS | `tests/test_experiment_registry.py`: 9 passed. |

## Key Findings

1. `scripts/experiments/lorenz/gen_maco_res.py`
- Issue: Duplicate legacy execution block remains at module scope.
- Impact: Script is not cleanly refactored and may execute unexpected duplicated logic.
- Suggested fix: Remove the trailing duplicate block and keep only the `if __name__ == '__main__':` workflow.

2. `scripts/run_all.py`
- Issue: Mixed tabs/spaces and invalid `else` placement.
- Impact: Top-level orchestrator fails to run (`TabError`).
- Suggested fix: Normalize indentation and repair control flow.

3. Legacy config imports still present
- `scripts/experiments/logmaps/config_logmapres.py`
- `scripts/experiments/tentmaps/config_tentmapres.py`
- `scripts/experiments/lorenz/config_lorenzres.py`
- `scripts/experiments/noise_length/config_noise_length.py`
- `scripts/experiments/lorenz_hypertune/htune_config.py`
- Issue: Still tied to older `scripts.config` path constants.
- Suggested fix: Refactor to consume central `CONFIG_*` dicts via `get_config` + `resolve_paths`, or retire obsolete files.

4. Lorenz hypertune partial migration
- Files affected: `genres_pca_htune.py`, `genres_ica_htune.py`, `genres_dca_htune.py`, `genres_sfa_htune.py`.
- Issue: Scripts parse `--config` and load `cfg`, but output writes still go through imported `interim_save_path` from `htune_config.py`.
- Suggested fix: Replace `interim_save_path` usage with `cfg['paths']['interim_res_path']`.

5. Dummy combine script not aligned
- File: `scripts/experiments/dummy_experiment/Z_combine_final_res.py`
- Issue: Not using config-loader path model.
- Suggested fix: Either fully refactor to match family combine scripts or explicitly exclude dummy family from refactor requirements.

## Verified Working Parts

- `scripts/experiments/run_family.py`
  - Accepts `--config`
  - Resolves absolute path via `Path(args.config).resolve()`
  - Passes config path to subprocess command builder

- Family scripts (`gen_*.py`, `genres_*.py`, `maco_noise.py`, `maco_length.py`)
  - All targeted files include `--config`, `get_config(...)`, and `resolve_paths(...)`
  - Main guard pattern present

## Conclusion

The refactor is close but not yet complete/correct end-to-end. The remaining blockers are concentrated in:
- top-level orchestrator validity (`scripts/run_all.py`),
- legacy helper/config modules,
- Lorenz MaCo duplicate code,
- Lorenz hypertune output path propagation.
