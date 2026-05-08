import numpy as np

from scripts.config import project_path


interim_res_path = project_path / 'results/interim/noise_length'
iterim_res_path = interim_res_path
final_res_path = project_path / 'results/final/noise_length'

### ----------------- ###
# length-dependence params
### ----------------- ###

Ls = list(range(100, 1_000, 200)) + list(range(1_000, 3_001, 1_000))
length_params = dict(
    nvars=3,
    N=10,  # number of realizations
    Ls=Ls,
    n=max(Ls),  # Length of time series
    rint=(3.8, 4.),  # interval to chose from the value of r parameter
    A0=np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]]),  # basic connection structure
    n_epochs=100,
    n_models=10,
    dx=1,
    dy=2,
    dz=1,
    nh=20,  # number of hidden units
    tau=1,
    batch_size=500,
    trainset_size=80,
    testset_size=10,
    validset_size=10,
    lr=1e-2,  # learning rate
)

#---------------------------------#
# Noise dependence params
#---------------------------------#
Ls = 10. ** np.arange(-3, .5, 0.25)
noise_params = dict(
    nvars=3,
    N=10,  # number of realizations
    Ls=Ls,  # noise levels
    n=1_000,  # Length of time series
    rint=(3.8, 4.),  # interval to chose from the value of r parameter
    A0=np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]]),  # basic connection structure
    n_epochs=100,
    n_models=10,
    tau=1,
    dx=1,
    dy=2,
    dz=1,
    nh=20,  # number of hidden units
    batch_size=500,
    trainset_size=80,
    testset_size=10,
    validset_size=10,
    lr=1e-2,  # learning rate
)
