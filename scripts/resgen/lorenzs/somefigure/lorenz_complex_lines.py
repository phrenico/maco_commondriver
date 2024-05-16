import numpy as np
import matplotlib.pyplot as plt
from matplotlib import lines, patches
from argparse import Namespace
from mpl_toolkits.mplot3d import Axes3D

def set_axislabel(ax, subscript=""):
    ax.set_xlabel(r'$x_{}$'.format(subscript))
    ax.set_ylabel(r'$y_{}$'.format(subscript))
    ax.set_zlabel(r'$z_{}$'.format(subscript))

v = np.load('../../../../data/lorenz/lorenz_0_long.npz.npy', allow_pickle=True)

T = 300_000  # 400_000
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

kwargs = dict(lw=0.2)

im1 = ax1.plot(*Z.T, **kwargs)
im2 = ax2.plot(*X.T, **kwargs)
im3 = ax3.plot(*Y.T, **kwargs)


# ax2.view_init(elev=0, azim=0)
# ax3.view_init(elev=0, azim=0)


axs = [ax1, ax2, ax3]


# plt.colorbar(im1, ax=axs, orientation='horizontal', pad=0.1, label='Driver')


for i, ax in enumerate(axs):
    set_axislabel(ax, i+1)

transFigure = fig.transFigure.inverted()

arrow1 = patches.FancyArrowPatch(
    (0.5, 0.6),  # posB
    (0.4, .5),
    shrinkA=0,  # so tail is exactly on posA (default shrink is 2)
    shrinkB=0,  # so head is exactly on posB (default shrink is 2)
    transform=fig.transFigure,
    color="black",
    arrowstyle="-|>",  # "normal" arrow
    mutation_scale=20,  # controls arrow head size
    linewidth=2,
)
arrow2 = patches.FancyArrowPatch(
    (0.7, 0.6),  # posB
    (0.8, .5),
    shrinkA=0,  # so tail is exactly on posA (default shrink is 2)
    shrinkB=0,  # so head is exactly on posB (default shrink is 2)
    transform=fig.transFigure,
    color="black",
    arrowstyle="-|>",  # "normal" arrow
    mutation_scale=20,  # controls arrow head size
    linewidth=2,
)
fig.patches.append(arrow1)
fig.patches.append(arrow2)

plt.savefig('Lorenz_complex_lines_sameview.png', transparent=True)
ax1.view_init(elev=0, azim=0)

plt.savefig('Lorenz_complex_lines.png', transparent=True)
# plt.close()

# fig2 = plt.figure(figsize=(15, 10))
# a = plt.subplot(111, projection='3d')
# im3 = a.scatter(*Z.T, c=z, cmap='jet',alpha=1)
# set_axislabel(a)
#
# plt.savefig('Lorenz_colored_long_Z.png')
# plt.show()