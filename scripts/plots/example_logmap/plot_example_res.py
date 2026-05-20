"""This figure contain the results of prediction and reconstruction for the example logistic map.
the figure consists of 6 subplots

A B C
D E F

A: Learing curves for the prediction task
B: reconstructed example time series section
C: Legend if needed or about the architecture
D: correlogram for the prediction task
E: correlogram for the reconstruction task
F: correlogram of the prediction and reconstruction task coefficient of determinations
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import minmax_scale, scale

from matplotlib.lines import Line2D

from scripts.config_runall import example_logmap_final_res_path
from scripts.plots.config_figgen import fig_path


res_path = example_logmap_final_res_path

def move_figure(f, x, y):
    """Move figure's upper left corner to pixel (x, y)"""
    try:
        manager = f.canvas.manager
        window = getattr(manager, 'window', None)
        if window is None:
            return

        backend = plt.get_backend()
        if backend == 'TkAgg' and hasattr(window, 'wm_geometry'):
            window.wm_geometry("+%d+%d" % (x, y))
        elif backend == 'WXAgg' and hasattr(window, 'SetPosition'):
            window.SetPosition((x, y))
        elif hasattr(window, 'move'):
            # Works on many QT/GTK backends.
            window.move(x, y)
    except Exception:
        # Positioning is optional; never fail plotting because of backend/window differences.
        pass


def plot_example_res():
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))    

    plot_learning_prediction(ax_learn=axs[0, 0],
                             ax_valid_loss=axs[0, 1],
                             ax_predictions=axs[0, 2])
    
    plot_coefreconstruct(ax_reconstruct=axs[1, 2])

    plot_ts_reconstruct(ax_ts=axs[1, 0],
                        ax_rec=axs[1, 1])

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # write A B C ...
    for i, ax in enumerate(axs.ravel()):
        ax.text(-0.1, 1.1, chr(65+i, ), transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='top')
    

    fig_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path / 'example_logmap_res.png')
    # move_figure(fig, 0, 0)
    # plt.show()

def plot_learning_prediction(ax_learn, ax_predictions, ax_valid_loss):
    global res_path
    df = pd.read_csv(res_path / 'mappercoach_res.csv')
    learnings = np.load(res_path / 'learning_curves.npy')
    valid_loss = np.load(res_path / 'valid_loss.npy')
    x_pred = df['x_pred'].values
    x_test = df['x_test'].values

    # find best model
    ind_best_model = np.argmin(valid_loss)

    # Compute correlations
    r = np.corrcoef(x_test, x_pred, rowvar=False)[0, 1]

    # cluster the endpoints of learning curves into 2 clusters
    nc = 2
    km = KMeans(n_clusters=nc, random_state=2)
    clusts = 1 - km.fit_predict(learnings[-1:, :].T)

    ax1 = ax_learn
    ax2 = ax_predictions

    clust_cols = ['#F9A448', 'b']

    for i in range(nc):
        _ = ax1.plot(learnings[:, i==clusts], color=clust_cols[i])

    ax2.plot(minmax_scale(x_test), minmax_scale(x_pred), '.', alpha=1., color='#F71616')
    ax2.plot([0, 1], [0, 1], 'k--',)

    


    ax1.set_xlabel('# epochs')
    ax1.set_ylabel(R'$L$ (mean squared loss)')
    ax1.set_yscale('log')
    ax1.set_xscale('log')



    ax2.text(0.05, .9, r'$r^2={:.3f}$'.format(r**2), transform=ax2.transAxes)
    ax2.set_xlabel(r"$x(t)$")
    ax2.set_ylabel(r"$\hat{x}(t)$")
    ax2.set_xlim(0,1)
    ax2.set_ylim(0,1)



    custom_lines = [Line2D([0], [0], color=clust_cols[1], lw=2),
                    Line2D([0], [0], color=clust_cols[0], lw=2)]
    ax1.legend(custom_lines, ['cluster 1', 'cluster 2'], loc='lower left')



    # fig.tight_layout(pad=1, h_pad=0, w_pad=1)

    ax3 = ax_valid_loss # plt.axes((0.25, 0.7, 0.2, 0.2))
    barcols = [clust_cols[i] for i in clusts]
    ax3.bar(range(len(valid_loss)), valid_loss, color=barcols)
    ax3.plot(ind_best_model, valid_loss[ind_best_model]+0.01, 'k*', ms=3)
    # ax3.set_yticklabels([0, 0.05, 0.1])
    ax3.set_yscale('log')
    ax3.set_xticklabels([])
    ax3.set_xticks([])
    ax3.set_xlabel('models')
    ax3.yaxis.set_label_position("left")
    ax3.set_ylabel('validation loss')


def plot_ts_reconstruct(ax_ts, ax_rec):
    # Load data
    df = pd.read_csv(res_path / 'mappercoach_res.csv')
    cc_pred= df['cc_pred'].values
    cc_val = df['cc_test'].values


    # compute correlation
    rec_perform = np.corrcoef(cc_val[:], cc_pred[:])
    # print(rec_perform)

    cp = scale(cc_pred)
    c = scale(cc_val)

    T = 59


    axs2 = [ax_ts, ax_rec]

    axs2[0].plot(minmax_scale(c[:T]), label="original", color='k', alpha=1, lw=2)
    axs2[0].plot(minmax_scale(np.sign(rec_perform[0, 1]) * cp[:T]), label="reconstructed",
                linestyle='-', color='#9EE004', alpha=1, lw=1.5)

    axs2[1].plot(scale(cc_val[:]), scale(np.sign(rec_perform[0, 1]) * cc_pred[:]), '.', alpha=1., color="#9EE004")
    axs2[1].plot([-1.9, 1.5], [-1.9, 1.5], 'k--')
    axs2[1].text(.05, .9, r'$r^2={:.2f}$'.format(rec_perform[0, 1]**2), transform=axs2[1].transAxes)


    axs2[0].set_xlabel(r'$t$ (simulation step)')
    axs2[0].set_ylabel(r'$z$')

    axs2[0].legend(loc="lower left")


    axs2[1].set_xlabel(r'normalized $z(t-1)$')
    axs2[1].set_ylabel(r'normalized $\hat{z}(t-1)$')
    axs2[1].set_xlim([-1.9, 1.5])
    axs2[1].set_ylim([-1.9, 1.5])


def plot_coefreconstruct(ax_reconstruct):

    rdf = pd.read_csv(res_path / 'r_values.csv', index_col=0)
    rs = rdf.values
    rsq = rs**2


    p = np.polyfit(rs[:, 0]**2, rs[:, 1]**2, deg=1)
    x = np.arange(0.85, 1, 0.001)
    y = np.polyval(p, x)
    
    nclust = 2
    gm_model = GaussianMixture(n_components=nclust, random_state=0).fit(rsq[:, :1])
    gmres = gm_model.predict(rsq[:, :1])
    gmeans = gm_model.means_
    gcovs = gm_model.covariances_

    meta_r = np.corrcoef(rsq[gmres==1].T)

    axin_xlim = [0.955, 0.999]
    axin_ylim = [0.80, 0.99]

    ax = ax_reconstruct

    # axin = ax.inset_axes([0.2, 0.45, 0.47, 0.47])

    cols = ['b', '#F9A448'][::-1]
    cluster_names = ['1', '2'][::-1]
    means = []
    stdevs = []
    for i in range(nclust):
        v = rsq[gmres==i]
        ax.plot(v[:, 0], v[:, 1], '.', color=cols[i], ms='10', label='cluster {}'.format(cluster_names[i]))
        # axin.plot(v[:, 0], v[:, 1], '.', color=cols[i], ms='8')
        means.append(v.mean(axis=0))
        stdevs.append(v.std(axis=0))

    # ax.legend()
    ax.set_xlabel(r"$r_\mathrm{prediction}^2$")
    ax.set_ylabel(r"$r_\mathrm{reconstruction}^2$")

    ax.set_ylim(-.1, 1)
    ax.set_xlim(0.65, 1)
    # axin.set_xlim(axin_xlim)
    # axin.set_ylim(axin_ylim)
    # ax.indicate_inset_zoom(axin, edgecolor="black")


def main():

    plot_example_res()


    pass

if __name__ == '__main__':
    main()
