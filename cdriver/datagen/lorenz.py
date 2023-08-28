import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint
from collections import OrderedDict
from tqdm import tqdm

def dfds(u, t, sigma1, rho1, beta1, sigma2, rho2, beta2, sigma3, rho3, beta3, kappa):
    """Derivative of 3 coupled Lorenz system

    :param u: state
    :param t: time
    :param sigma1: sigma parameter for the 1st system
    :param rho1: rho parameter for the 1st system
    :param beta1: beta parameter for the 1st system
    :param sigma2: sigma parameter for the 2nd system
    :param rho2: rho parameter for the 2nd system
    :param beta2: beta parameter for the 2nd system
    :param sigma3: sigma parameter for the 3rd system
    :param rho3: rho parameter for the 3rd system
    :param beta3: beta parameter for the 3rd system
    :param kappa: coupling coefficients
    :return: derivative of the state
    """
    x, y, z, x2, y2, z2, x3, y3, z3 = u
    kappa12, kappa21, kappa13, kappa31, kappa23, kappa32 = kappa[0, 1], kappa[1, 0], kappa[0, 2], kappa[2, 0], kappa[1, 2], kappa[2, 1]

    dx = sigma1 * ((y - x) + kappa12 * (y2 - x)) + kappa13 * (y3 - x)
    dy = x * (rho1 - z) - y
    dz = x * y - beta1 * z

    dx2 = sigma2 * ((y2 - x2) + kappa21 * (y - x2)) + kappa23 * (y3 - x2)
    dy2 = x2 * ((rho2 + 0.1) - z2) - y2
    dz2 = x2 * y2 - (beta2 + 0.03) * z2

    dx3 = sigma3 * ((y3 - x3) + kappa31 * (y - x3)) + kappa32 * (y2 - x3)
    dy3 = x3 * ((rho3 + 0.1) - z3) - y3
    dz3 = x3 * y3 - (beta3 + 0.03) * z3

    return [dx, dy, dz, dx2, dy2, dz2, dx3, dy3, dz3]

if __name__=="__main__":
    # Data Generation
    np.random.seed(312)
    dt = 1e-3
    t = np.arange(0, 1000, dt)
    sigma = 10
    rho = 27
    beta = 8 / 3

    for i in tqdm(range(100)):
        # Unidirectional coupling
        kappa = np.zeros([3, 3])
        alpha = 0.1
        kappa[1, 0] = alpha + (1-alpha) * np.random.rand(1)[0]
        kappa[2, 0] = alpha + (1-alpha) * np.random.rand(1)[0]

        param_dict = OrderedDict(sigma1=sigma, rho1=rho, beta1=beta,
                                 sigma2=sigma+np.random.normal(0, 2, 1),
                                 rho2=rho+np.random.normal(0, 2, 1),
                                 beta2=beta+np.random.normal(0, 0.2, 1),
                                 sigma3=sigma+np.random.normal(0, 2, 1),
                                 rho3=rho+np.random.normal(0, 2, 1),
                                 beta3=beta+np.random.normal(0, 0.2, 1),
                                 kappa=kappa)
        params = tuple(param_dict.values())

        v0 = (10 * np.random.rand(9)).tolist()

        v = odeint(dfds, v0, t, (params))

        #save data
        ds = 200
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

