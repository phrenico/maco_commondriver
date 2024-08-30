import sys
sys.path.append('/home/phrenico/Projects/Codes/maco_commondriver')
from scripts.config import project_path

# Logmap output path
interim_res_path = project_path  / 'results/interim/lorenzs'
final_res_path = project_path / 'results/final'
figure_path = project_path / 'figures'

# common  parameters
data_path_template = '../../../data/lorenz/lorenz_{}.npz'
N = 50
train_split = 0.5