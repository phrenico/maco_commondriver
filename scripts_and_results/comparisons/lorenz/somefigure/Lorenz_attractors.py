import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

v = np.load('../../../../data/lorenz/lorenz_0_long.npz.npy', allow_pickle=True)
T = 70000
Z = v[:T:5, :3]
X = v[:T:5, 3:6]
Y = v[:T:5, 6:9]


print(v.shape)


# Plot Lorenz attractors in 3d
plt.figure()
plt.subplot(111, projection='3d')
plt.plot(*Z.T, '-', lw=2, color='teal')
plt.title('Z')
plt.savefig('Z.png')

plt.figure()
plt.subplot(111, projection='3d')
plt.plot(*X.T, '-', lw=2, color='tab:orange')
plt.title('X')
plt.savefig('X.png')

plt.figure()
plt.subplot(111, projection='3d')
plt.plot(*Y.T, '-', lw=2, color='tab:blue')
plt.title('Y')
plt.savefig('Y.png')

plt.show()
