import numpy as np
import matplotlib.pyplot as plt
from argparse import Namespace
from mpl_toolkits.mplot3d import Axes3D

def set_axislabel(ax):
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_zlabel(r'$z$')

v = np.load('../../../../data/lorenz/lorenz_0_long.npz.npy', allow_pickle=True)

T = -1  # 400_000
ds = 20
Z = v[:T:ds, :3]
X = v[:T:ds, 3:6]
Y = v[:T:ds, 6:9]
z = Z[:, 1]


h = 0.55
w = 0.7
fig = plt.figure(figsize=(15, 10))

ax2 = fig.add_axes(rect=(0., 0., w, h), projection='3d')
ax3 = fig.add_axes(rect=(0.4, 0., w, h), projection='3d')
ax1 = fig.add_axes(rect=(0.2, 0.5, w+0.1, h+0.1), projection='3d', clip_on=False)

im1 = ax1.scatter(*Z.T, c=z, cmap='jet',alpha=1)
im2 = ax2.scatter(*X.T, c=z, cmap='jet',alpha=1)
im3 = ax3.scatter(*Y.T, c=z, cmap='jet',alpha=1)

ax1.view_init(elev=0, azim=0)
# ax2.view_init(elev=0, azim=0)
# ax3.view_init(elev=0, azim=0)


axs = [ax1, ax2, ax3]


# plt.colorbar(im1, ax=axs, orientation='horizontal', pad=0.1, label='Driver')


for ax in axs:
    set_axislabel(ax)




plt.savefig('Lorenz_complex.png', transparent=True)
# plt.close()

# fig2 = plt.figure(figsize=(15, 10))
# a = plt.subplot(111, projection='3d')
# im3 = a.scatter(*Z.T, c=z, cmap='jet',alpha=1)
# set_axislabel(a)
#
# plt.savefig('Lorenz_colored_long_Z.png')
# plt.show()