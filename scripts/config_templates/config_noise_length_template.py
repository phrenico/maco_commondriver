
# Standalone template config for a single NOISE_LENGTH experiment run

import numpy as np

CONFIG_NOISE_LENGTH = {
	'datagen': {
		'nvars': 3,
		'N':     1,
		'rint':  (3.8, 4.0),
		'A0': np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=float),
	},
	'length_sweep': {
		'Ls': list(range(100, 1000, 200)) + list(range(1000, 3001, 1000)),
		'n':  3000,
	},
	'noise_sweep': {
		'Ls': [10.0 ** x for x in [-3, -2.75, -2.5, -2.25, -2, -1.75, -1.5, -1.25, -1, -0.75, -0.5, -0.25, 0, 0.25, 0.5]],
		'n':  1000,
	},
	'preprocessing': {
		'trainset_size': 80,
		'testset_size':  10,
		'validset_size': 10,
	},
	'paths': {
		'final_res_path': 'paper_artifacts/results/final/noise_length_demo',
		'figure_path':    'paper_artifacts/figures/noise_length_demo',
	},
	'maco': {
		'n_epochs':   100,
		'n_models':   10,
		'batch_size': 500,
		'lr':         1e-2,
		'dx':         1,
		'dy':         2,
		'dz':         1,
		'n_hidden':   20,
		'tau':        1,
	},
}
