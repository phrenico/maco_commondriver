'''Script to run PCA on logistic map data-set
1. Generate data
2. Run PCA
3. Plot results
4. Save results

'''
import numpy as np

from sklearn.decomposition import PCA
from data_generators import LogmapExpRunner, time_delay_embedding, comp_ccorr, get_maxes, save_results, train_test_split
import matplotlib.pyplot as plt

from tqdm import tqdm

if __name__=="__main__":

    #1. Generate data
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


    #2. Run PCA
    d_embed = 3
    maxcs = []
    for i in tqdm(range(N)):
        X = time_delay_embedding(dataset[i][:, 1], delay=1, dimension=d_embed)
        Y = time_delay_embedding(dataset[i][:, 2], delay=1, dimension=d_embed)
        z = dataset[i][d_embed-1:, 0]

        X_train, Y_train, z_train, X_test, Y_test, z_test = train_test_split(X, Y, z, train_split)

        D = np.concatenate([X_train, Y_train], axis=1)

        model = PCA(n_components=1).fit(D)
        zpred = model.transform(np.concatenate([X_test, Y_test], axis=1))

        maxcs.append(get_maxes(*comp_ccorr(z_test, zpred[:, 0]))[1])

    # Save results
    df = save_results(fname='./pca_res.csv', r=maxcs, N=N, method='PCA', dataset='logmap_fixed')

    #3. Plot results
    plt.figure()
    plt.hist(maxcs)
    plt.show()