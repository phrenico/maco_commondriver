from scripts.config import project_path

# Logmap output path
interim_res_path = project_path  / 'results/interim/lorenz'
final_res_path = project_path / 'results/final'
figure_path = project_path / 'figures'

# common  parameters
data_path_template = str(project_path / 'data/lorenz/lorenz_{}.npz')
N = 50
train_split = 1. / 3
valid_split = 1. / 3
test_split = 1. / 3