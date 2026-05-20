
# Standalone template config for a single LORENZ_HTUNE experiment run

CONFIG_LORENZ_HTUNE = {
	'sweep': {
		'ns_components': [2],
	},
	'preprocessing': {
		'train_split': 0.5,
		'valid_split': 0.25,
	},
	'paths': {
		'interim_res_path': 'paper_artifacts/results/interim/lorenz_htune_demo',
		'final_res_path':   'paper_artifacts/results/final/lorenz_htune_demo',
		'figure_path':      'paper_artifacts/figures/lorenz_htune_demo',
	},
	'data': {
		'N': 1,
		'data_path_template': 'data/lorenz/lorenz_0.npz',
	},
}
