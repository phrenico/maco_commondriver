import argparse
from pathlib import Path

import pandas as pd

from scripts.experiments.config_loader import get_config, resolve_paths
from scripts.experiments.lorenz_hypertune.htune_config import get_htune_paths, plot_htune

_REPO_ROOT = Path(__file__).resolve().parents[3]


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', default='scripts/config_runall.py')
	args = parser.parse_args()
	cfg = resolve_paths(get_config('lorenz_htune', args.config), _REPO_ROOT)
	paths = get_htune_paths(cfg)

	df = pd.read_csv(paths['interim_res_path'] / 'dca_htune.csv', index_col=0)
	plot_htune(df, 'DCA', save=True, path=paths['figure_path'])