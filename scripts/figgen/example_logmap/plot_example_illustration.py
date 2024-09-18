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
import networkx as nx

import sys
sys.path.append('./')
sys.path.append('../../')
from scripts.figgen.config_figgen import fig_path

# matplotlib.style.use('./figure_onecol_config.mplstyle')


def plot_returnmap(ax):
    def f(z,  r=3.99):
        '''dynamics of the driver
        '''
        return np.abs(r * z * (1 - z))

    def g(y, z, r=3.99, beta=0.2):
        '''dynamics of the forced system
        '''
        return r * y *(1 - y - beta * z)

    print("Generate Figure 4. (logmap_returnmap.eps)")

    # res_path = Path('../../../results/final/example_logmap')

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
    newcmp = ListedColormap(newcolors,
                            name='OrangeBlue')

    ax0 = ax
    ax1 = ax.inset_axes([1.1, 0.1, 0.1, 0.8])

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
    cb1.set_ticklabels([0, 0.5, 1])

    ax0.set_xlabel("$y_{t-1}$", labelpad=-8)
    ax0.set_ylabel("$y_{t}$", labelpad=-8)
    ax0.set_yticks([0, 1])
    ax0.set_xticks([0, 1])

def plot_markov_ts(ax):
    # draw a small markov chain o -> o
    pos = {0:(-1,1), 1:(0, 1), 2:(1, 1),
           3:(-1,0), 4:(0, 0), 5:(1, 0),
           6:(-1,-1), 7:(0, -1),8:(1, -1)}
    color_x = 'tab:blue'
    color_y = 'tab:blue'
    color_z = 'teal'
    cols = {0:color_x, 1:color_x, 2:color_x,
            3:color_z, 4:color_z, 5:color_z,
            6:color_y, 7:color_y, 8:color_y}
    node_labels = {0:'$x_{t-1}$', 1:'$x_t$', 2:'$x_{t+1}$',
                3:'$z_{t-1}$', 4:'$z_t$', 5:'$z_{t+1}$',
                6:'$y_{t-1}$', 7:'$y_t$', 8:'$y_{t+1}$'}
    ns = 2_000

    G = nx.DiGraph()

    G.add_edges_from([(0, 1), (1, 2),
                    (3, 4), (4, 5),
                    (6, 7), (7, 8),
                    (3, 1), (3, 7),
                    (4, 2), (4, 8)])



    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=ns,
                        node_color=list(cols.values()), alpha=1)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='k', node_size=ns)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=16, labels=node_labels)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_axis_off()


def plot_dynamics(ax):
    ns = 2_000
    pos = {0:(-1,2), 1:(0, 2), 2:(1, 2),
           3:(0,0), 4:(1, 0)}
    node_labels = {0: '$S_{t-1}$', 1: '$S_t$', 2: '$S_{t+1}$',
                   3: '$Y_{t-1}$', 4: '$Y_t$'}
    edge_labels = {(0, 1): "$F$",
                (1, 2): "$F$",
                (0, 3): "$\Phi$",
                (3, 0): "$\Phi^{-1}$",
                (3, 4): "$\widetilde{F}$",
                (2, 4): "$g$"}
    edge_list = list(edge_labels.keys())
    straight_edges_list = [edge_list[i] for i in [0, 1,-1, -2]]
    curved_edges_list = [edge_list[i] for  i in [2, 3]]

    c1 = 'tab:green'
    c2 = 'tab:blue'
    cols = {0:c1, 1:c1, 2:c1,
            3:c2, 4:c2}

    con_styles = ['arc3,rad=0.',
                'arc3,rad=0.25',]

    # draw second plot
    G = nx.MultiDiGraph()

    G.add_edges_from([(0, 1), (1, 2),
                    (0, 3), (3, 0),
                    (3, 4), (2, 4)])



    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=ns, node_color=list(cols.values()), alpha=1)
    # draw straight edges
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='k', node_size=ns, edgelist=straight_edges_list,
                        arrowstyle='-|>', connectionstyle=con_styles[0])
    # draw curved edges
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='k', node_size=ns, edgelist=curved_edges_list,
                        arrowstyle='-|>', connectionstyle=con_styles[1])

    nx.draw_networkx_labels(G, pos, ax=ax, font_size=16, labels=node_labels)


    # add text for som labels
    text_kwargs = {'fontsize':16, 'horizontalalignment':'center', 'verticalalignment':'center'}
    ax.text(-0.5, pos[0][1]+.2, '$F$', **text_kwargs)
    ax.text(0.5, pos[1][1]+.2, '$F$', **text_kwargs)
    ax.text(1.1, (pos[2][1] + pos[4][1]) / 2, '$g$', **text_kwargs)
    ax.text(0.5, pos[3][1]+.2, '$\widetilde{F}$', **text_kwargs, color='red')
    ax.text(-0.9, 0.5, '$\Phi^{-1}$', **text_kwargs, color='k')
    ax.text(-0.1, (pos[2][1] + pos[4][1]) / 2, '$\Phi$', **text_kwargs, color='k')

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.2, 3.)
    ax.set_axis_off()


def plot_markov_dynamics_returnmap():
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    plot_markov_ts(ax[0])

    plot_dynamics(ax[1])

    plot_returnmap(ax[2])

    fig.tight_layout()
    # plot A B C
    ax[0].text(-0.05, 1., 'A', fontsize=20, fontweight='bold', transform=ax[0].transAxes)
    ax[1].text(-0.05, 1., 'B', fontsize=20, fontweight='bold', transform=ax[1].transAxes)
    ax[2].text(-0.15, 1., 'C', fontsize=20, fontweight='bold', transform=ax[2].transAxes)


    figManager = plt.get_current_fig_manager()
    # move the window to the top left Tkinter
    # figManager.window.wm_geometry("+0+0")
    # figManager.window.showMaximized()
    # plt.show()
    fig.savefig(fig_path / 'markov_dynamics_returnmap.png', bbox_inches='tight', dpi=300)

def main():
    plot_markov_dynamics_returnmap()
    

if __name__ == '__main__':
    main()