import numpy as np
import matplotlib.pyplot as plt
from argparse import Namespace

v = np.load('../../../../data/lorenz/lorenz_0_long.npz.npy', allow_pickle=True)

T = -1  # 400_000
ds = 20
Z = v[:T:ds, :3]
X = v[:T:ds, 3:6]
Y = v[:T:ds, 6:9]
z = Z[:, 1]

fig = plt.figure(figsize=(15, 10))
ax1 = plt.subplot(131, projection='3d')
ax2 =  plt.subplot(132, projection='3d')
ax3 = plt.subplot(133, projection='3d')



ax1 = plt.subplot(131, projection='3d')
im1 = ax1.scatter(*X.T, c=z, cmap='jet',alpha=1)
# plt.colorbar(im1)


im2 = ax2.scatter(*X.T, c=z, cmap='jet', alpha=1)
ax2.view_init(elev=120)
# plt.colorbar(im2)


im3 = ax3.scatter(*X.T, c=z, cmap='jet', alpha=1)
ax3.view_init(elev=180)
fig.tight_layout()

axs = [ax1, ax2, ax3]

def set_axislabel(ax):
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_zlabel(r'$z$')

for ax in axs:
    set_axislabel(ax)

plt.colorbar(im3, ax=axs, orientation='horizontal', pad=0.1, label='Driver')


plt.savefig('Lorenz_colored_long.png')
plt.close()

fig2 = plt.figure(figsize=(15, 10))
a = plt.subplot(111, projection='3d')
im3 = a.scatter(*Z.T, c=z, cmap='jet',alpha=1)
set_axislabel(a)

plt.savefig('Lorenz_colored_long_Z.png')
plt.show()