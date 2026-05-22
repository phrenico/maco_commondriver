# Experiment Config Tutorial

This tutorial shows how to use a config file to customize and run a single experiment in the maco_commondriver framework.

## 1. Pick a Template

Copy the template for your experiment family from `scripts/config_templates/`:

- `config_logmaps_template.py`
- `config_tentmaps_template.py`
- `config_lorenz_template.py`
- `config_lorenz_htune_template.py`
- `config_example_logmap_template.py`
- `config_noise_length_template.py`
- `config_comparison_plots_template.py`

Example (for logmaps):
```bash
cp scripts/config_templates/config_logmaps_template.py my_logmaps_config.py
```

## 2. Customize the Standalone Dict

Open your copied file (e.g. `my_logmaps_config.py`) and edit values directly inside the `CONFIG_<FAMILY>` dict.

Example (`my_logmaps_config.py`):

```python
import numpy as np

CONFIG_LOGMAPS = {
	'datagen': {
		'N': 1,
		'n': 1000,
		'rint': (3.8, 4.0),
		'A0': np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=float),
		'A':  np.array([[1.0, 0.0, 0.0], [0.3, 1.0, 0.0], [0.4, 0.0, 1.0]], dtype=float),
	},
	'preprocessing': {
		'train_split': 1.0 / 3,
		'valid_split': 1.0 / 3,
		'd_embed': 3,
	},
	'paths': {
		'interim_res_path': 'paper_artifacts/results/interim/logmaps_demo',
		'final_res_path':   'paper_artifacts/results/final/logmaps_demo',
		'figure_path':      'paper_artifacts/figures/logmaps_demo',
	},
	# ... keep the rest of the keys required by this family
}
```

## 3. Run the Experiment with Your Config

Use the `--config` argument to point to your config file. For example, to run the logmaps family:

```bash
cdriver-run-family logmaps --config my_logmaps_config.py
```

Or for a dry run (no computation, just prints the planned steps):

```bash
cdriver-run-family logmaps --dry-run --config my_logmaps_config.py
```

## 4. Output

Results will be written to the paths you set in your config (e.g. `paper_artifacts/results/interim/logmaps_demo`).

For family-owned plots, set `paths.figure_path` to control where plot images are saved.
This applies to `example_logmap`, `noise_length`, `lorenz_htune`, and the standalone
plot entrypoints for `logmaps`, `tentmaps`, and `lorenz`.

For the shared aggregate comparison figure, use a dedicated `CONFIG_COMPARISON_PLOTS`
config and always pass it explicitly with `--config` when calling
`python -m scripts.plots.plot_comparisons`.

## 5. Notes

- Keep each template standalone: do not rely on imports from another config file.
- If a datagen field is used as a NumPy array at runtime, keep it as `np.array(...)` in the template.
- The config file must define a `CONFIG_<FAMILY>` dict matching the experiment family you run.
- The shared comparison plot is stricter: it requires `CONFIG_COMPARISON_PLOTS` in an explicit external config file and does not run without `--config`.
- For more details, see `README.md` and the template files in `scripts/config_templates/`.
