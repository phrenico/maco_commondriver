from scripts.config import figures_root, lorenz_data_path_template, lorenz_final_res_path, lorenz_interim_res_path, lorenz_realizations

# Logmap output path
interim_res_path = lorenz_interim_res_path
final_res_path = lorenz_final_res_path
figure_path = figures_root


# create directories if they don't exist
interim_res_path.mkdir(parents=True, exist_ok=True)
final_res_path.mkdir(parents=True, exist_ok=True)
figure_path.mkdir(parents=True, exist_ok=True)

# common  parameters
data_path_template = lorenz_data_path_template
N = lorenz_realizations
train_split = 1. / 3
valid_split = 1. / 3
test_split = 1. / 3