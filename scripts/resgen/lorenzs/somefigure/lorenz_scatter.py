import numpy as np
import matplotlib.pyplot as plt
from argparse import Namespace
from sklearn.preprocessing import scale

from mpl_toolkits.mplot3d import Axes3D

q = np.load('./results_0.npz', allow_pickle=True)

print(q.files)
a = Namespace(**q)

print(a.X.shape, a.z_test.shape)


fig = plt.figure(figsize=(15, 10))
ax1 = plt.subplot(131, projection='3d')
ax2 =  plt.subplot(132, projection='3d')
ax3 = plt.subplot(133, projection='3d')



ax1 = plt.subplot(131, projection='3d')
im1 = ax1.scatter(*a.X.T, c=a.z, cmap='jet',alpha=1)
# plt.colorbar(im1)


im2 = ax2.scatter(*a.X.T, c=a.z, cmap='jet', alpha=1)
ax2.view_init(elev=120)
# plt.colorbar(im2)


im3 = ax3.scatter(*a.X.T, c=a.z, cmap='jet', alpha=1)
ax3.view_init(elev=180)
fig.tight_layout()

axs = [ax1, ax2, ax3]

def set_axislabel(ax):
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_zlabel(r'$z$')

for ax in axs:
    set_axislabel(ax)

plt.colorbar(im3, ax=axs, orientation='horizontal', pad=0.1, label='Shared Driver')


plt.savefig('Lorenz_colored.png')
plt.show()