import numpy as np
import matplotlib.pyplot as plt



fig, ax = plt.subplots(10, 10, sharex=True, sharey=True)
for i in range(100):
    data_path = '../../../data/lorenz/lorenz_{}.npz'.format(i)
    data = np.load(data_path)

    X = data['v']

    ax[i//10, i%10].plot(X[:, [3, 6]], lw=1, alpha=0.5)

plt.tight_layout(rect=[0, 0, 1, 1], h_pad=0, w_pad=0, pad=0)
plt.show()