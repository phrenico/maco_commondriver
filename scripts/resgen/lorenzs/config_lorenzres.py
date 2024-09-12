import sys
sys.path.append('../../../')
sys.path.append('./')
from scripts.config import project_path

print("Project path: ", project_path)

# Logmap output path
interim_res_path = project_path  / 'results/interim/lorenzs'
final_res_path = project_path / 'results/final'
figure_path = project_path / 'figures'

# common  parameters
data_path_template = str(project_path) + '/data/lorenz/lorenz_{}.npz'
N = 50
train_split = 0.5
valid_split = 0.25
test_split = 0.25