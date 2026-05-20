
# Standalone template config for a single EXAMPLE_LOGMAP experiment run

CONFIG_EXAMPLE_LOGMAP = {
	'datagen': {
		'N': 1,
		'n': 2000,
		'rint': (3.8, 4.0),
		'A0': [[0, 0, 0], [1, 0, 0], [1, 0, 0]],
		'A':  [[1.0, 0.0, 0.0], [0.3, 1.0, 0.0], [0.4, 0.0, 1.0]],
	},
	'preprocessing': {
		'trainset_size': 80,
		'testset_size':  10,
		'validset_size': 10,
	},
	'paths': {
		'final_res_path': 'paper_artifacts/results/final/example_logmap_demo',
	},
	'maco': {
		'n_epochs':   2000,
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
