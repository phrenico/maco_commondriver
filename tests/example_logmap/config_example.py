
# Standalone template config for a single EXAMPLE_LOGMAP experiment run
import numpy as np

CONFIG_EXAMPLE_LOGMAP = {
	'datagen': {
		'N': 1,
		'n': 10_000,
		'rint': (3.8, 4.0),
		'A0': np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]]),
		'A':  np.array([[1.0, 0.0, 0.0], [0.3, 1.0, 0.0], [0.4, 0.0, 1.0]]),
	},
	'preprocessing': {
		'trainset_size': 80,
		'testset_size':  10,
		'validset_size': 10,
	},
	'paths': {
		'final_res_path': '/home/zsiga/Projects/Codes/maco_commondriver/tests/example_logmap/results/',
		'figure_path': '/home/zsiga/Projects/Codes/maco_commondriver/tests/example_logmap/results/',
	},
	'maco': {
		'n_epochs':   10_000,
		'n_models':   10,
		'batch_size': 1000,
		'lr':         1e-2,
		'dx':         1,
		'dy':         2,
		'dz':         1,
		'n_hidden':   20,
		'tau':        1,
		'device':     'cpu',
	},
}
