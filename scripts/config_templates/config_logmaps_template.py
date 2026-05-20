
# Standalone template config for a single LOGMAPS experiment run

CONFIG_LOGMAPS = {
	'datagen': {
		'N': 1,
		'n': 1000,
		'rint': (3.8, 4.0),
		'A0': [[0, 0, 0], [1, 0, 0], [1, 0, 0]],
		'A':  [[1.0, 0.0, 0.0], [0.3, 1.0, 0.0], [0.4, 0.0, 1.0]],
	},
	'preprocessing': {
		'train_split': 1.0 / 3,
		'valid_split': 1.0 / 3,
		'd_embed': 3,
	},
	'paths': {
		'interim_res_path': 'paper_artifacts/results/interim/logmaps_demo',
		'final_res_path':   'paper_artifacts/results/final/logmaps_demo',
	},
	'maco': {
		'n_epochs':   300,
		'n_models':   10,
		'batch_size': 1000,
		'lr':         1e-2,
		'dx':         1,
		'dy':         2,
		'dz':         1,
		'n_hidden':   20,
		'tau':        1,
	},
	'methods': {
		'pca':    {'n_components': 1},
		'kpca':   {'n_components': 1, 'kernel': 'rbf'},
		'ica':    {'n_components': 2},
		'cca':    {'n_components': 1},
		'dcca':   {'features': [3, 3], 'layers': [20, 20, 1]},
		'dca':    {'n_components': 1, 'T': 5, 'n_init': 10},
		'sfa':    {'n_components': 1, 'poly_degree': 2},
		'shrec':  {'d_embed': 3},
		'anisom': {'d_embed': 3, 'd_grid': 2, 'sizes': [40, 20], 'epochs': 4},
		'random': {},
	},
}
