---
description: "Verify the --config CLI refactoring is complete and correct across all experiment families. Checks for missing argparse, old imports, path resolution, and propagation through orchestrators."
name: "Check --config Refactor"
agent: "agent"
tools: [grep_search, file_search, read_file, run_in_terminal]
---

You are a verification agent. Your job is to audit whether the `--config /path/to/file.py` CLI refactoring is fully and correctly implemented across the `maco_commondriver` codebase.

## What the refactoring should achieve

Every experiment script under `scripts/experiments/` must:
1. Accept an optional `--config <path>` CLI argument via `argparse`
2. Load config via `get_config('<family_key>', args.config)` from `scripts.experiments.config_loader`
3. Resolve paths via `resolve_paths(cfg, _REPO_ROOT)` where `_REPO_ROOT = Path(__file__).resolve().parents[N]`
4. No longer import from old per-family config modules (`config_lorenzres`, `config_logmapres`, `config_tentmapres`, `noise_params`, `length_params`, `logmapgen_params`, etc.)

Orchestrators must:
- `scripts/experiments/run_family.py`: accept `--config`, resolve to absolute path with `Path(args.config).resolve()`, pass to every subprocess command
- `scripts/run_all.py`: accept `--config`, resolve to absolute path, propagate to all `run_family` subprocesses

## Checks to perform

### 1. Central config infrastructure
- Confirm `scripts/experiments/config_loader.py` exists and exports `get_config`, `resolve_paths`
- Confirm `scripts/experiments/config.py` exists and defines all required dicts:
  `CONFIG_LOGMAPS`, `CONFIG_TENTMAPS`, `CONFIG_LORENZ`, `CONFIG_EXAMPLE_LOGMAP`, `CONFIG_NOISE_LENGTH`, `CONFIG_LORENZ_HTUNE`

### 2. Per-script argparse check
For every `gen_*.py`, `genres_*.py`, `maco_noise.py`, `maco_length.py` under `scripts/experiments/`, verify:
- Contains `--config` in an `argparse.ArgumentParser` block
- Calls `get_config(` with the appropriate family key
- Calls `resolve_paths(`
- Is wrapped in `if __name__ == '__main__':` (or uses `main()` called from it)

Use `grep_search` to check for `--config`, `get_config`, `resolve_paths` across `scripts/experiments/**/*.py`.

### 3. Z_combine_final_res.py scripts
Each family's combine script must use `cfg['paths']['final_res_path']` (or `cfg['paths']['interim_res_path']`) rather than old `scripts.config.*` attribute imports.

### 4. Old import elimination
Search for any remaining references to old config modules:
```
config_lorenzres|config_logmapres|config_tentmapres
logmapgen_params|logmapexamplegen_params
noise_params, final_res_path|length_params, final_res_path
from scripts.config import.*_res_path
```
Report any files that still use these patterns.

### 5. Path resolution correctness
For a sample of files (one from each family), read them and verify:
- `_REPO_ROOT` is computed with the correct number of `.parents[N]` (should reach the repo root, not a subdirectory)
- `resolve_paths` is called before accessing `cfg['paths']` values
- Paths stored in `config.py` are relative strings (not absolute)

### 6. Orchestrator propagation
Read `scripts/experiments/run_family.py` and confirm:
- `--config` argument is added to `argparse`
- `config_path = str(Path(args.config).resolve())` (absolute conversion)
- `config_path` is passed to `build_command(step, config_path=config_path)`

Read `scripts/run_all.py` and confirm:
- `--config` argument is added to `argparse`
- `config_path = str(Path(args.config).resolve())` (absolute conversion)
- `config_path` is passed to the `run_family` subprocess command

### 7. Dry-run smoke test
Run:
```bash
cd /home/zsiga/Projects/Codes/maco_commondriver
python -m scripts.experiments.run_family logmaps --dry-run --config /tmp/fake_config.py 2>&1 | head -20
python -m scripts.experiments.run_family lorenz --dry-run --config /tmp/fake_config.py 2>&1 | head -10
python -m scripts.experiments.run_family noise_length --dry-run --config /tmp/fake_config.py 2>&1 | head -10
```
Confirm `--config /tmp/fake_config.py` appears in the printed subprocess commands.

### 8. Registry tests
Run:
```bash
cd /home/zsiga/Projects/Codes/maco_commondriver
python -m pytest tests/test_experiment_registry.py -v 2>&1
```
All 9 tests must pass.

## Output format

Produce a structured report with sections:

**PASS / FAIL summary table** — one row per check above.

For any FAIL, include:
- The file path(s) with the issue
- The specific problem (e.g., "missing `--config` argparse", "still imports config_lorenzres")
- A suggested fix

If all checks pass, end with:
> ✅ All checks passed. The --config refactoring is complete and correct.
