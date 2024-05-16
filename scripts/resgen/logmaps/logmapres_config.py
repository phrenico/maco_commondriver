""" configuration for results on ht logmap dataset"""
from pathlib import Path

# project path
project_path = Path('/home/phrenico/Projects/Codes/maco_commondriver')

# Logmap output path
interim_res_path = project_path  / 'results/interim/logmaps'
final_res_path = project_path / 'results/final'
figure_path = project_path / 'figures'

# common training parameters
train_split = 0.5