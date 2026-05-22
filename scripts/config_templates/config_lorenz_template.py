
# Standalone template config for a single LORENZ experiment run

CONFIG_LORENZ = {
	'data': {
		'N': 1,
		'data_path_template': 'data/lorenz/lorenz_0.npz',
	},
	'preprocessing': {
		'train_split': 1.0 / 3,
		'valid_split': 1.0 / 3,
	},
	'paths': {
		'interim_res_path': 'paper_artifacts/results/interim/lorenz_demo',
		'final_res_path':   'paper_artifacts/results/final/lorenz_demo',
		'figure_path':      'paper_artifacts/figures/lorenz_demo',
	},
	'maco': {
		'n_epochs':   200,
		'n_models':   10,
		'batch_size': 1000,
		'lr':         1e-2,
		'dx':         3,
		'dy':         3,
		'dz':         1,
		'n_hidden':   20,
		'tau':        1,
	},
	'methods': {
		'pca':    {'n_components': 5},
		'ica':    {'n_components': 5},
		'cca':    {'n_components': 1, 'max_iter': 500},
		'dcca':   {'d_embed': 3, 'features': [3, 3], 'layers': [20, 20, 1]},
		'dca':    {'n_components': 5, 'T': 5, 'n_init': 10},
		'sfa':    {'n_components': 3, 'poly_degree': None},
		'shrec':  {'d_embed': 3},
		'random': {},
	},
}
