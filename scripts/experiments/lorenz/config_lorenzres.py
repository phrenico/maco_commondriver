from scripts.config import project_path

# Logmap output path
interim_res_path = project_path  / 'paper_artifacts/results/interim/lorenz'
final_res_path = project_path / 'paper_artifacts/results/final'
figure_path = project_path / 'paper_artifacts/figures'


# create directories if they don't exist
interim_res_path.mkdir(parents=True, exist_ok=True)
final_res_path.mkdir(parents=True, exist_ok=True)
figure_path.mkdir(parents=True, exist_ok=True)

# common  parameters
data_path_template = str(project_path / 'data/lorenz/lorenz_{}.npz')
N = 50
train_split = 1. / 3
valid_split = 1. / 3
test_split = 1. / 3