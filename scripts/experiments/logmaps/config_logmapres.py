""" configuration for results on ht logmap dataset"""
from scripts.config import figures_root, logmaps_final_res_path, logmaps_interim_res_path

# Logmap output path
interim_res_path = logmaps_interim_res_path
final_res_path = logmaps_final_res_path
figure_path = figures_root

# create directories if they don't exist
interim_res_path.mkdir(parents=True, exist_ok=True)
final_res_path.mkdir(parents=True, exist_ok=True)
figure_path.mkdir(parents=True, exist_ok=True)

# common training parameters
train_split = 1. / 3
valid_split = 1. / 3
test_split = 1 - (train_split + valid_split)