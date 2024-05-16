import numpy as np
import matplotlib.pyplot as plt
from argparse import Namespace
from sklearn.preprocessing import scale

from mpl_toolkits.mplot3d import Axes3D

q = np.load('./results_0.npz', allow_pickle=True)

a = Namespace(**q)

print(a.X.shape)

# Plot a 3D plot of the Lorenz attractor
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(*a.X.T, '.')
# ax.plot(*a.Y.T, '.')


plt.figure()
plt.subplot(111)
plt.plot(scale(a.z_test), label='original')
plt.plot(scale(a.z_pred), label='reconstructed')
plt.text(0, 2.7, r'$r^2 = {:.2f}$'.format(a.c[0]))

plt.legend(loc='upper right')
plt.savefig('reconstruct_ts.png')

plt.figure()
plt.subplot(111)
plt.plot(a.z_test[1:], a.z_pred, '.')

plt.show()