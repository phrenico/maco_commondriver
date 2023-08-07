'''Run experiments with SFA

'''
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
import dca
# from dca import DynamicalComponentsAnalysis as
DCA = dca.DynamicalComponentsAnalysis
from data_generators import LogmapExpRunner, comp_ccorr, get_maxes, train_test_split, save_results, time_delay_embedding

if __name__ == "__main__":
    # 1. Generate data
    N = 50  # number of realizations
    n = 20_000  # Length of time series
    rint = (3.8, 4.)  # interval to chose from the value of r parameter
    A0 = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]])  # basic connection structure
    A = np.array([[1., 0., 0.],
                  [0.3, 1., 0.],
                  [0.4, 0., 1.]])
    train_split = 0.5

    dataset = [LogmapExpRunner(nvars=3, baseA=A0, r_interval=rint).gen_experiment(n=n, A=A, seed=i)[0] for i in
               tqdm(range(N))]
    params = [LogmapExpRunner(nvars=3, baseA=A0, r_interval=rint).gen_experiment(n=2, A=A, seed=i)[1] for i in range(N)]

    d_embed = 3
    maxcs = []
    for i in tqdm(range(N)):
        data = dataset[i]

        z = data[:-(d_embed - 1), 0]
        X = time_delay_embedding(data[:, 1], dimension=d_embed)
        Y = time_delay_embedding(data[:, 2], dimension=d_embed)

        X_train, Y_train, z_train, X_test, Y_test, z_test = train_test_split(X, Y, z, train_split)
        D_train = np.concatenate([X_train, Y_train], axis=1)
        D_test = np.concatenate([X_test, Y_test], axis=1)

        # 2. Run SFA
        dca_model = DCA(d=1, T=5, n_init=10)
        dca_model.fit(D_train)
        zpred = dca_model.transform(D_test).squeeze()

        tau, c = comp_ccorr(zpred, z_test)
        maxcs.append(get_maxes(tau, c)[1])

    # Save results
    df = save_results(fname='./dca_res.csv', r=maxcs, N=N, method='DCA', dataset='logmap_fixed')

    # 3. Plot results
    plt.figure()
    plt.hist(maxcs)
    plt.show()