'''Script for generating the figure about the return map of the logistic map
## Contents

0. Imports & function definitions

1. Compute returnmap for different values for $z$
    1.1. Define values of $y$ and $z$ to compute the returnmap

    1.2. Compute values

2. Draw the figure
'''
import matplotlib.style
# 0. Imports & function definitions
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
from matplotlib.colors import ListedColormap

matplotlib.style.use('./figure_onecol_config.mplstyle')

def f(z,  r=3.99):
    '''dynamics of the driver
    '''
    return np.abs(r * z * (1 - z))

def g(y, z, r=3.99, beta=0.2):
    '''dynamics of the forced system
    '''
    return r * y *(1 - y - beta * z)


# 1. Compute returnmap for different values for z
# 1.1. Define values of y and z to compute the returnmap on
y = np.arange(0, 1, 0.001)
zs = np.arange(0, 1, 0.05)

# 1.2. Compute values
y_tp = np.array([g(y, z) for z in zs]).T
y_tpp = np.array([g(g(y, z), f(z)) for z in zs]).T


# 2. Draw the figure

# Custom colormap
top = cm.get_cmap('Greens_r', 256)
bottom = cm.get_cmap('Reds', 256)
newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                       bottom(np.linspace(0, 1, 128))))
newcmp = ListedColormap(newcolors, name='OrangeBlue')

fig, [ax0, ax1] = plt.subplots(1, 2, gridspec_kw={'width_ratios': [8, 1]})

# _ = [ax0.plot(y, y_tp[:, i], 'b', alpha=1, color=cm.plasma(zs[i])) for i in range(y_tp.shape[1])]
_ = [ax0.plot(y, y_tp[:, i], 'b', alpha=1, color=newcmp(zs[i])) for i in range(y_tp.shape[1])]


ax0.set_xlim(0, 1)
ax0.set_ylim(0, 1)


# cmap = cm.plasma
cmap = newcmp
norm = mpl.colors.Normalize(vmin=0, vmax=0)

cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,
                                norm=norm,
                                orientation='vertical')

cb1.set_label('driver value $z_{t-1}$')

cb1.set_ticks([-0.1, 0, 0.1])

ax0.set_xlabel("$y_{t-1}$", labelpad=-8)
ax0.set_ylabel("$y_{t}$", labelpad=-8)
ax0.set_yticks([0, 1])
ax0.set_xticks([0, 1])

# fig.suptitle('Return Map of the Forced Logistic Map')

# fig.tight_layout(pad=0, w_pad=1, rect=[0, 0, 1, 1])

fig.savefig('./resfigure/logmap_returnmap.eps')
fig.savefig('./resfigure/logmap_returnmap.png')
# plt.show()