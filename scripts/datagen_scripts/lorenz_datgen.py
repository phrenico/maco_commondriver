import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint
from collections import OrderedDict
from tqdm import tqdm
from types import SimpleNamespace
from cdriver.datagen.lorenz import dfds

from datagen_config import lorenzgen_params




if __name__=="__main__":

    dparams = SimpleNamespace(**lorenzgen_params)

    # Data Generation
    np.random.seed(dparams.rseed)
    dt = dparams.dt
    t = dparams.t
    sigma = dparams.sigma
    rho = dparams.rho
    beta = dparams.beta

    ds = dparams.ds  # down-sample rate before saving the data

    for i in tqdm(range(dparams.N)):
        # Unidirectional coupling
        kappa = np.zeros([3, 3])
        alpha = 0.1
        kappa[1, 0] = alpha + (1-alpha) * np.random.rand(1)[0]
        kappa[2, 0] = alpha + (1-alpha) * np.random.rand(1)[0]

        param_dict = OrderedDict(sigma1=sigma, rho1=rho, beta1=beta,
                                 sigma2=sigma+np.random.normal(0, 2, 1)[0],
                                 rho2=rho+np.random.normal(0, 2, 1)[0],
                                 beta2=beta+np.random.normal(0, 0.2, 1)[0],
                                 sigma3=sigma+np.random.normal(0, 2, 1)[0],
                                 rho3=rho+np.random.normal(0, 2, 1)[0],
                                 beta3=beta+np.random.normal(0, 0.2, 1)[0],
                                 kappa=kappa)
        params = tuple(param_dict.values())

        v0 = (10 * np.random.rand(9)).tolist()

        v = odeint(dfds, v0, t, (param_dict, ))
        
        if i == 0:
            np.save('../../data/lorenz/lorenz_{}_long.npz'.format(i), v)
            # exit()

        #save data
        param_dict['ds'] = ds
        np.savez('../../data/lorenz/lorenz_{}.npz'.format(i), v=v[::ds], t=t[::ds], params=param_dict)

    # # Plotting
    # fig = plt.figure(figsize=(10, 10))
    # ax1 = fig.add_subplot(131, projection='3d')
    # ax1.plot(v[:, 0], v[:, 1], v[:, 2])
    #
    # ax2 = fig.add_subplot(132, projection='3d')
    # ax2.plot(v[:, 3], v[:, 4], v[:, 5])
    #
    # ax3 = fig.add_subplot(133, projection='3d')
    # ax3.plot(v[:, 6], v[:, 7], v[:, 8])
    #
    # plt.show()

