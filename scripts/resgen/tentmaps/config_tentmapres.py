""" configuration for results on the tentmap dataset"""
from scripts.config import project_path

# Logmap output path
interim_res_path = project_path  / 'results/interim/tentmaps'
final_res_path = project_path / 'results/final'
figure_path = project_path / 'figures'

# common training parameters
train_split = 0.5
valid_split = 0