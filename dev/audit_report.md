# Repository Audit Report — maco_commondriver

**Date:** 2026-05-20  
**Auditor:** DeepSeek TUI (v0.8.39)  
**Repository:** `maco_commondriver` — "Reconstructing shared dynamics with a deep neural network"

---

## 1. Executive Summary

The repository is a well-structured research codebase accompanying two papers on hidden common-driver reconstruction. The overall architecture — a core package (`cdriver/`), isolated `uv` environments per method family (`envs/`), a declarative experiment registry, and a unified configuration system — is a strong foundation. The recent configuration refactoring (May 2026) cleaned up much technical debt.

However, the audit uncovered **several broken import paths in plot scripts**, significant code duplication across ~30 experiment scripts, a few hardcoded-path leftovers, a misconfigured `uv` workspace block, and gaps in test coverage and CI. Most issues are fixable with modest effort; the critical ones are the broken plot imports (Section 2.1) and the uv workspace misconfiguration (Section 2.2).

---

## 2. Critical Issues (Will Cause Runtime Failures)

### 2.1 Broken Imports in Plot Scripts (4–5 files)

The config refactoring removed path constants that several plot scripts still import. These scripts will fail with `ImportError` at startup.

| File | Missing Import | Fix |
|------|---------------|-----|
| `scripts/plots/config_figgen.py:1` | `noise_length_final_res_path` | Does not exist in `scripts.config_runall` |
| `scripts/plots/logmaps/comparison_plot_logmap.py:5` | `logmaps_final_res_path` | Does not exist in `scripts.config_runall` |
| `scripts/plots/tentmaps/comparison_plot_tentmap.py:5` | `tentmaps_final_res_path` | Does not exist in `scripts.config_runall` |
| `scripts/plots/example_logmap/plot_example_res.py:25` | `example_logmap_final_res_path` | Does not exist in `scripts.config_runall` |
| `scripts/plots/lorenz/comparison_plot_lorenz.py:5` | `misc_figure_path` | Exists in `config_runall` — this one is OK |

All missing names should be derived from `final_results_root` + the respective subdirectory, or restored as explicit `Final[Path]` constants in `scripts/config_runall.py`.

### 2.2 Malformed `uv` Workspace Configuration

`pyproject.toml` lines 87–101:

```toml
[tool.uv.workspace]
members = [
    "envs/sfa_env/envs/sfa_env",   # nonexistent nested path
    "envs/sfa_env",
    ".",                             # root listed multiple times
    "envs/dca_env",
    ".",                             # duplicate
    "envs/dcca_env",
    ".",                             # duplicate
    "envs/shrec_env",
    ".",                             # duplicate
    "envs/maco_env",
    ".",                             # duplicate
]
```

The root `.` appears 5 times; the phantom path `envs/sfa_env/envs/sfa_env` does not exist. This likely causes `uv sync` warnings or errors. The block should be reduced to:

```toml
[tool.uv.workspace]
members = [
    ".",
    "envs/maco_env",
    "envs/dca_env",
    "envs/dcca_env",
    "envs/shrec_env",
    "envs/sfa_env",
]
```

---

## 3. Reproducibility Gaps

### 3.1 Unpinned Git Dependencies

Two environments use unpinned Git URLs:

- `envs/dca_env/pyproject.toml`: `dynamicalcomponentsanalysis = { git = "https://github.com/BouchardLab/DynamicalComponentsAnalysis" }`
- `envs/shrec_env/pyproject.toml`: `shrec = { git = "https://github.com/williamgilpin/shrec" }`

Neither specifies a `rev`, `tag`, or `branch`. Any future commit to these repos can silently break the build. Add a `rev = "..."` or `tag = "..."` to each source.

### 3.2 `rseed` Bug in `datagen_config.py`

```python
# scripts/datagen_scripts/datagen_config.py:41
lorenzgen_params = dict(rseed=np.random.seed(312), ...)
```

`np.random.seed(312)` returns `None`, so `lorenzgen_params['rseed']` is `None`, not `312`. However, the seed is set as a side effect on the global RNG. The intent was probably `rseed=312`. Fix:

```python
lorenzgen_params = dict(rseed=312, ...)
np.random.seed(312)
```

### 3.3 No Seed Configuration in Config Dicts

The `CONFIG_LOGMAPS`, `CONFIG_TENTMAPS`, etc. dicts do not expose a `seed` or `random_state` field. Seeds are hardcoded inside the generation loops (e.g., `seed=i`). This means:
- Running a subset of realizations gives different data than the first N of a full run.
- There's no way to fix the global seed for a full pipeline from the config file.

### 3.4 Lorenz Data Generation — Duplicate Entry Points

Two scripts can generate Lorenz data:

| File | Quality |
|------|---------|
| `scripts/datagen_scripts/lorenz_datgen.py` | **Canonical** — uses config, proper path resolution |
| `cdriver/datagen/lorenz.py` (`__main__` block) | **Legacy** — hardcoded `'../../data/lorenz/...'` relative paths, different RNG logic |

The `cdriver/datagen/lorenz.py` `__main__` block should be removed or replaced with a comment pointing to the canonical script. Its `dfds()` function and the Lorenz ODE are needed by `lorenz_datgen.py`, so the file should stay — just the `if __name__ == "__main__"` block should be cleaned.

### 3.5 Hardcoded Path in `cdriver/datagen/tent_map.py`

```python
# cdriver/datagen/tent_map.py, __main__ block
save_fname = '../../data/tent_1d_data.csv'
```

This relative path will only work if invoked from `cdriver/datagen/`. Remove or redirect to the proper config-driven path.

---

## 4. Code Quality Issues

### 4.1 Massive Code Duplication in Experiment Scripts

The experiment scripts under `scripts/experiments/{logmaps,tentmaps,lorenz}/` follow an identical pattern for each method. For example, compare `gen_pca_res.py` and `gen_ica_res.py` in the Lorenz family — they differ in exactly 3 lines (the model class, method name, and result filename). This pattern repeats across ~30 files.

**Impact:** Changing the common scaffolding (e.g., adding progress logging, error handling, seed control) requires editing each file individually.

**Recommendation:** Create a single parameterized runner per family that accepts the model class, method name, and config keys. The current per-method scripts can remain as thin wrappers (or be retired entirely in favor of registry-driven execution).

### 4.2 Debug `print()` Left in Library Code

```python
# cdriver/evaluate/evalz.py:14
def eval_lin(X, Y):
    ...
    print(regmod.score(X, Y), regmod2.score(Y, X))  # debug print
```

Remove or guard with `logging.debug()`.

### 4.3 Confusing `AniSOM` Comment

```python
# cdriver/network/anisom.py:13
class AniSOM(nn.Module):
    def __init__(self, space_dim, grid_dim, sizes):
        """Anisotropic Self-Organizing Map BUT IT IS NOT WORKING YET!!!!!!!!!!1 ?????? What did I mean with that?
```

This module corresponds to a _published_ paper (Benkő et al., Neural Networks, 2026). The comment should explain the current status of the implementation relative to the paper.

### 4.4 `global device` Anti-Pattern in Lorenz MaCo Script

```python
# scripts/experiments/lorenz/gen_maco_res.py:22-24
def preprocess(X, Y):
    global device   # fragile
    ...
```

The `device` variable is set at module level inside `if __name__ == '__main__'`. When this script is imported as a module (which the runner does), `device` will be `None` or undefined. The `preprocess` override should accept `device` as a closure argument or the model's own `self.device`.

### 4.5 `torchvision` as Root Dependency

`pyproject.toml` lists `torchvision` under `dependencies`, but no code in `cdriver/` uses `torchvision` beyond `transforms.ToTensor()` and `transforms.Compose()`, which are actually in `torch` (not `torchvision`). Installing `cdriver` pulls in a large GPU stack unnecessarily for users who only run PCA/ICA baselines. Remove `torchvision` from root dependencies and add it only where needed (already handled per-environment).

### 4.6 Docstring Mismatch in `cdriver/datagen/control.py`

```python
def shuffle_phase(x):
    """shuffles the phase of the signal in Fourier domain

    :param x: signal
    :param sf: sampling rate     # parameter not in signature
    ...
```

Remove the `sf` param from the docstring.

### 4.7 Typo: `min_c_strenght` → `min_c_strength`

In `cdriver/datagen/tent_map.py`:

```python
def sample_params(self, A=None, a=None, x0=None, min_c_strenght=0.1, seed=None):
```

The parameter `min_c_strenght` (missing 'h' in 'strength') propagates through the TentMap experiment runner. Fix the typo (or add a deprecation wrapper if external code depends on the old name).

### 4.8 Inconsistent API: `SimpleNamespace` vs Raw `dict`

- `gen_logmapdata()` uses `SimpleNamespace(**param_dict)` to access params as attributes.
- `gen_tentmapdata()` uses raw dict access: `tentmapgen_params['N']`.

This causes confusion about which calling convention to use. Standardize on one approach (recommend `SimpleNamespace` for read-only config access, or plain dicts consistently).

---

## 5. Testing Gaps

### 5.1 Coverage Summary

| Area | Tested? | Notes |
|------|---------|-------|
| Package imports | Yes (`test_import.py`) | Minimal |
| Time-delay embedding | Yes (`test_tde.py`) | 3 test cases |
| AniSOM shapes | Yes (`test_anisom.py`) | Smoke test only |
| Registry consistency | Yes (`test_experiment_registry.py`) | 9 good tests |
| Execution paths | Yes (`test_execution_paths.py`) | Dry-run, save, combine |
| MaCo (Mapper-Coach) | No | Core algorithm — no unit tests |
| LogMap generator | No | No tests for boundary conditions, dataset shape |
| Lorenz ODE | No | No tests |
| TentMap generator | No | No tests |
| Evaluation metrics | No | `comp_ccorr`, `get_maxes`, `eval_lin` untested |
| Savers | No | `save_results` partially tested via execution path test |
| Splitters | No | `train_test_split`, `train_valid_test_split` untested |
| Experiment scripts | No | No integration tests with small N |

### 5.2 No CI/CD

There are no GitHub Actions workflows, no `tox.ini`, no `pre-commit` config. The `.github/` directory contains only a prompt artifact. Adding a minimal CI pipeline (lint + test on PR) would catch regressions like the broken plot imports.

---

## 6. Structural / Documentation Issues

### 6.1 README Minor Discrepancies

- The README says `data/` is a "placeholder for generated datasets" but `data/` is not tracked in git (no `.gitkeep`). The directory is created at import time by `config_runall.py`, which works but is unusual — a `.gitkeep` would make the intent clearer.
- The `cdriver-run-family` console script is documented but it wraps `scripts/experiments/run_family.py`; users might also try `python -m scripts.experiments.run_family` directly, which works identically. This is fine but worth noting.

### 6.2 `paper_artifacts/` Contains Binary Model Files

```
paper_artifacts/results/final/example_logmap/best_model.pth   (PyTorch model)
paper_artifacts/results/final/example_logmap/models.pkl        (pickle)
```

Binary files tracked in git inflate the repository size and can't be diffed. Consider moving these to a release asset or DVC/Git LFS. At minimum, document their purpose in `paper_artifacts/README.md`.

---

## 7. Action Plan

### Phase 1: Fix Critical Breakage (Estimated: 1–2 hours)

| Priority | Task | Files Affected |
|----------|------|---------------|
| P0 | Fix broken plot imports — add missing path constants to `scripts/config_runall.py` | `scripts/config_runall.py`, `scripts/plots/config_figgen.py`, `scripts/plots/logmaps/comparison_plot_logmap.py`, `scripts/plots/tentmaps/comparison_plot_tentmap.py`, `scripts/plots/example_logmap/plot_example_res.py` |
| P0 | Fix `uv` workspace `members` in root `pyproject.toml` | `pyproject.toml` |
| P0 | Fix `rseed=None` bug in `datagen_config.py` | `scripts/datagen_scripts/datagen_config.py` |

### Phase 2: Reproducibility Hardening (Estimated: 2–4 hours)

| Priority | Task |
|----------|------|
| P1 | Pin git dependency revisions in `envs/dca_env/pyproject.toml` and `envs/shrec_env/pyproject.toml` |
| P1 | Remove legacy `__main__` blocks from `cdriver/datagen/lorenz.py` and `cdriver/datagen/tent_map.py` |
| P1 | Add `seed` / `random_state` fields to all `CONFIG_*` dicts and wire them into datagen/experiment scripts |
| P1 | Create `data/.gitkeep` for clarity |

### Phase 3: Code Quality (Estimated: 3–6 hours)

| Priority | Task |
|----------|------|
| P2 | Remove debug `print()` from `cdriver/evaluate/evalz.py` |
| P2 | Fix docstring mismatch in `cdriver/datagen/control.py` |
| P2 | Fix `min_c_strenght` typo in `cdriver/datagen/tent_map.py` |
| P2 | Standardize config access (dict vs. SimpleNamespace) across datagen modules |
| P2 | Remove `torchvision` from root `pyproject.toml` dependencies |
| P2 | Clean up `AniSOM.__init__` docstring comment |
| P3 | Refactor `global device` in Lorenz MaCo script to use closure or model attribute |

### Phase 4: Reduce Duplication (Estimated: 4–8 hours)

| Priority | Task |
|----------|------|
| P3 | Create a parameterized `MethodRunner` class that accepts model class, method name, and config keys |
| P3 | Replace ~30 per-method experiment scripts with thin wrappers (or retire them if registry can dispatch directly) |
| P3 | Consolidate `build_series_loaders` vs. `preprocess` override pattern across families |

### Phase 5: Testing & CI (Estimated: 4–8 hours)

| Priority | Task |
|----------|------|
| P3 | Add unit tests for `MaCo` (forward pass shapes, loss computation, training loop with tiny data) |
| P3 | Add unit tests for datagen modules (LogMap, TentMap, Lorenz ODE) |
| P3 | Add unit tests for evaluation metrics (`comp_ccorr`, `get_maxes`, `eval_lin`) |
| P3 | Add integration test: run one realization of each family end-to-end with small N |
| P3 | Add `.github/workflows/test.yml` — lint (flake8/ruff) + pytest on push/PR |

---

## 8. Repository Strengths

Despite the issues above, the repository has several commendable design choices:

- **Clean separation of concerns** — `cdriver/` is a proper importable package; experiment orchestration lives in `scripts/`.
- **`uv` workspace with isolated environments** — each method family has its own dependency set, avoiding the "monolithic requirements.txt" problem common in research code.
- **Declarative experiment registry** (`experiment_registry.py`) — adding a new method means registering it once; the runner, combiner, and plotter pick it up automatically.
- **Unified config system** with external override via `--config` — allows quick parameter sweeps without editing source.
- **Dry-run mode** — lets users verify the pipeline graph before running computation.
- **Config templates + tutorial** — lowers the barrier for new users.
- **Good test coverage for the registry** — the 9 registry tests enforce consistency between declared methods, result files, and UV environments.
- **Git-tracked paper artifacts** — final CSVs and figures are versioned alongside code.

---

## 9. Summary

| Category | Count | Severity |
|----------|-------|----------|
| Broken imports (crashes) | 4 | Critical |
| Malformed config | 1 | Critical |
| Reproducibility gaps | 4 | High |
| Code quality (debug prints, typos, anti-patterns) | 7 | Medium |
| Duplication | ~30 files | Medium |
| Testing gaps | ~8 modules uncovered | Medium |
| Missing CI | 1 | Low |

The repository is in good shape for a research codebase and the recent refactoring effort shows awareness of software engineering practices. Phase 1 fixes should be applied immediately; Phases 2–5 can be scheduled incrementally.
