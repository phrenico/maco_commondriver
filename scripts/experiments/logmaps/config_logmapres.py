""" configuration for results on ht logmap dataset"""
from scripts.config import project_path

# Logmap output path
interim_res_path = project_path  / 'results/interim/logmaps'
final_res_path = project_path / 'results/final'
figure_path = project_path / 'figures'

# common training parameters
train_split = 1. / 3
valid_split = 1. / 3
test_split = 1 - (train_split + valid_split)