""" Data Generation parameter configurations"""
import numpy as np

# Example Logistic map dataset configuration
logmapexamplegen_params = dict(N=1,  # number of realizations
                               n=10_000,  # Length of time series
                               rint=(3.8, 4.),  # interval to chose from the value of r parameter
                               A0=np.array([[0, 0, 0],
                                            [1, 0, 0],
                                            [1, 0, 0]]), # basic connection structure
                               A=np.array([[1., 0., 0.],
                                           [0.3, 1., 0.],
                                           [0.4, 0., 1.]])
                                )

# Logmaps dataset configuration
logmapgen_params = dict(N=50,  # number of realizations
                        # n=15_000,  # Length of time series
                        n=int(3*2_000),  # Length of time series
                        rint=(3.8, 4.),  # interval to chose from the value of r parameter
                        A0=np.array([[0, 0, 0],
                                     [1, 0, 0],
                                     [1, 0, 0]]),  # basic connection structure
                        A=np.array([[1., 0., 0.],
                                    [0.3, 1., 0.],
                                    [0.4, 0., 1.]]))  

# Tentmaps generation configuration
tentmapgen_params = dict(N=50,
                        #  n=15_000,
                        n=int(3*2_000),  # Length of time series
                        aint=(2, 10.),  # interval to chose from the value of alpha parameter
                        A0=np.array([[0, 0, 0],
                                     [1, 0, 0],
                                     [1, 0, 0]]))  # basic connection structure

# Lorenz dataset configuration
dt = 1e-3
lorenzgen_params = dict(rseed=np.random.seed(312),
                        N=50,  # nof realizations
                        dt=dt,  # time step
                        t=np.arange(0, 2000, dt),  # time axis
                        sigma=10,  # sigma parameter
                        rho=27,  # rho parameter
                        beta=8 / 3,  # beta parameter
                        ds=200)  # down-sample rate before saving the data
