from pathlib import Path
from typing import Final


project_path: Final[Path] = Path(__file__).resolve().parents[1]
data_root: Final[Path] = project_path / 'data'
artifacts_root: Final[Path] = project_path / 'paper_artifacts_smoketest'
results_root: Final[Path] = artifacts_root / 'results'
interim_results_root: Final[Path] = results_root / 'interim'
final_results_root: Final[Path] = results_root / 'final'
figures_root: Final[Path] = artifacts_root / 'figures'
misc_figure_path: Final[Path] = figures_root
lorenz_htune_figure_path: Final[Path] = figures_root / 'lorenz_htune'

lorenz_data_path: Final[Path] = data_root / 'lorenz'
lorenz_data_path_template: Final[str] = str(lorenz_data_path / 'lorenz_{}.npz')

logmaps_interim_res_path: Final[Path] = interim_results_root / 'logmaps'
tentmaps_interim_res_path: Final[Path] = interim_results_root / 'tentmaps'
lorenz_interim_res_path: Final[Path] = interim_results_root / 'lorenz'
lorenz_htune_interim_res_path: Final[Path] = interim_results_root / 'lorenz_htune'
noise_length_interim_res_path: Final[Path] = interim_results_root / 'noise_length'

logmaps_final_res_path: Final[Path] = final_results_root
tentmaps_final_res_path: Final[Path] = final_results_root
lorenz_final_res_path: Final[Path] = final_results_root
lorenz_htune_final_res_path: Final[Path] = final_results_root / 'lorenz_htune'
noise_length_final_res_path: Final[Path] = final_results_root / 'noise_length'
example_logmap_final_res_path: Final[Path] = final_results_root / 'example_logmap'


# Realization-count defaults. Edit these for quick tests or per-family runs.
example_logmap_realizations: Final[int] = 1
N = 1
logmap_realizations: Final[int] = N
tentmap_realizations: Final[int] = N
lorenz_realizations: Final[int] = N
lorenz_htune_realizations: Final[int] = N
noise_length_realizations: Final[int] = 1

for path in (
	data_root,
	artifacts_root,
	results_root,
	interim_results_root,
	final_results_root,
	figures_root,
	misc_figure_path,
	lorenz_htune_figure_path,
	logmaps_interim_res_path,
	tentmaps_interim_res_path,
	lorenz_interim_res_path,
	lorenz_htune_interim_res_path,
	noise_length_interim_res_path,
	lorenz_htune_final_res_path,
	noise_length_final_res_path,
	example_logmap_final_res_path,
):
	path.mkdir(parents=True, exist_ok=True)