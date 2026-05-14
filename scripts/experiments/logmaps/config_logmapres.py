""" configuration for results on ht logmap dataset"""
from scripts.config import project_path

# Logmap output path
interim_res_path = project_path  / 'paper_artifacts/results/interim/logmaps'
final_res_path = project_path / 'paper_artifacts/results/final'
figure_path = project_path / 'paper_artifacts/figures'

# create directories if they don't exist
interim_res_path.mkdir(parents=True, exist_ok=True)
final_res_path.mkdir(parents=True, exist_ok=True)
figure_path.mkdir(parents=True, exist_ok=True)

# common training parameters
train_split = 1. / 3
valid_split = 1. / 3
test_split = 1 - (train_split + valid_split)