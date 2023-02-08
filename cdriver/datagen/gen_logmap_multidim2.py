import numpy as np
import matplotlib.pyplot as plt
from common import gen_ts, vec_reflect
from pandas import DataFrame


import matplotlib.pyplot as plt

def logmap(x, r, A):
    return vec_reflect( r * x * (1  - np.dot(A, x)))

gen_logmap = gen_ts(logmap)

def main():
    np.random.seed(321)
    N = 4
    x0 = np.random.rand(N)
    n = 20000
    A = np.eye(N)
    A[2, 0] = 0.1
    A[2, 1] = 0.1
    A[3, 0] = 0.1
    A[3, 1] = 0.1
    logkwargs = dict(r=3.99*np.ones(N), A=A)

    x = gen_logmap(x0, n, logkwargs)

    # Save out results
    save_fname = '../../data/logmap_2d_data2.csv'
    x_df = DataFrame(x)
    x_df.to_csv(save_fname)

    # plt.plot(x)
    # plt.show()

    from cmfsapy.dimension.fsa import fsa
    from cdriver.preprocessing.tde import time_delay_embedding

    for j in range(x.shape[1]):
        mds = []
        ds = range(1, 7)
        for i in ds:
            U = time_delay_embedding(x[:, j], delay=1, dimension=i)
            d, r, inds = fsa(U, k=100)
            mds.append(np.nanmedian(d, axis=0))
        mds = np.array(mds).T
        print(mds)

        plt.figure()
        plt.plot(mds)
        plt.ylim([0, max(ds)])
        window = plt.get_current_fig_manager().window
        print(window.geometry())
        window.setGeometry(500*j, 500, 500, 500)

    plt.show()


if __name__=="__main__":
    main()