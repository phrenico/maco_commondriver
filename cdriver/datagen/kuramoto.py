import numpy as np
from sdeint import itoint
from functools import partial


def dtheta(x, t, w, K):
    dx = w - np.sum(K @ np.sin(np.subtract.outer(x, x)), axis=1)
    return dx

def G(x, t, s):
    return s * np.eye(len(x))


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.signal import coherence, periodogram

    np.random.seed(256)

    dt = 0.001  # sampling interval
    t = np.arange(0, 20, dt)  # time axis
    n_vars = 3  # number of variables

    w = 2 * np.pi * np.array([25, np.pi*13, 314 / np.pi ])

    print(w)


    K = np.zeros([n_vars, n_vars])
    K[1, 0] = 0
    K[2, 0] = 0

    x0 = np.random.rand(n_vars)

    f = partial(dtheta, w = w, K=K) # function to integrate
    noise_lev = 0  # noise level
    g = partial(G, s=noise_lev)  # noise term in the stochastic differential equation

    # x = odeint(dtheta, x0, t)
    x = itoint(f, g, x0, t)
    s = np.sin(x)

    # plt.plot(t, s[:, 0])
    # plt.xlim(0, 1)
    # plt.show()

    plt.figure()
    for i in range(n_vars):
            print(i, )
            u = s[:, i]

            f, c = periodogram(u, fs=1 / dt, nfft=256)

            plt.plot(f, c, label='{}'.format(i))
    plt.legend()
    # plt.yscale('log')

    plt.figure()

    for i in range(n_vars - 1):
        for j in range(i + 1, n_vars):
            print(i, j)
            u = s[:, i]
            v = s[:, j]

            f, c = coherence(u, v, fs=1 / dt, nperseg=1024)

            plt.plot(f, c, label='{}-{}'.format(i, j))
    plt.legend()

    plt.show()
