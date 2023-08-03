import numpy as np

from sklearn.cross_decomposition import CCA

from data_generators import LogmapExpRunner, time_delay_embedding, comp_ccorr, get_maxes, save_results, train_test_split
import matplotlib.pyplot as plt

from tqdm import tqdm
import pandas as pd

if __name__=="__main__":
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
    maxcs2 = []
    maxcs3 = []
    for i in tqdm(range(N)):
        data = dataset[i]

        z = data[:-(d_embed - 1), 0]
        X = time_delay_embedding(data[:, 1], dimension=d_embed)
        Y = time_delay_embedding(data[:, 2], dimension=d_embed)

        X_train, Y_train, z_train, X_test, Y_test, z_test = train_test_split(X, Y, z, train_split)

        cca = CCA(n_components=1)
        cca.fit(X_train, Y_train)

        zpred, zpred2 = cca.transform(X_test, Y_test)

        tau, c = comp_ccorr(zpred[:, 0], z_test)
        tau, c2 = comp_ccorr(zpred2[:, 0], z_test)
        tau, c3 = comp_ccorr((zpred[:, 0] + zpred2[:, 0]) / 2, z_test)

        maxcs.append(get_maxes(tau, c)[1])
        maxcs2.append(get_maxes(tau, c2)[1])
        maxcs3.append(get_maxes(tau, c3)[1])

cca_df = pd.DataFrame(np.array([range(N), maxcs3, N*['cca'], N*['logistic_fixed']]).T, columns=['data_id', 'r', 'method', 'dataset'])
cca_df.to_csv('./cca_res.csv')

plt.hist(maxcs3)
plt.show()