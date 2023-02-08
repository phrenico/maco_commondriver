import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.preprocessing import minmax_scale, scale
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.stats import norm



def learnforcast(df, learnings, test_loss):
    '''Generates the Figure about prediction performance

    :param df:
    :param learnings:
    :param test_loss:
    :return:
    '''
    x_pred = df['x_pred'].values
    x_valid = df['x_valid'].values
    # Y_test = df[["Y_1_valid", "Y_2_valid"]].values
    # cc_pred = df['cc_pred'].values
    # cc_val = df['cc_valid'].values

    # find best model
    ind_best_model = np.argmin(test_loss)

    # Compute correlations
    r = np.corrcoef(x_valid, x_pred, rowvar=False)[0, 1]
    # print(r**2)

    # cluster the endpoints of learning curves into 2 clusters
    nc = 2
    km = KMeans(n_clusters=nc, random_state=2)
    clusts = km.fit_predict(learnings[-1:, :].T)
    # print(np.sum(clusts==0), np.sum(clusts==1))

    # visualize reasults
    fig, axs = plt.subplots(1, 2)

    ax1 = axs[0]
    ax2 = axs[1]

    clust_cols = ['#F9A448', 'b']

    for i in range(nc):
        _ = ax1.plot(learnings[:, i == clusts], color=clust_cols[i])

    ax2.plot(minmax_scale(x_valid), minmax_scale(x_pred), '.', alpha=0.2, color='#F71616')
    ax2.plot([0, 1], [0, 1], 'k--', )

    ax1.set_xlabel('# epochs')
    ax1.set_ylabel(R'$L$ (mean squared loss)')
    ax1.set_yscale('log')
    ax1.set_xscale('log')

    ax2.text(0.05, .9, r'$r^2={:.3f}$'.format(r ** 2), transform=ax2.transAxes)
    ax2.set_xlabel(r"$x(t)$")
    ax2.set_ylabel(r"$\hat{x}(t)$")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    custom_lines = [Line2D([0], [0], color=clust_cols[1], lw=2),
                    Line2D([0], [0], color=clust_cols[0], lw=2)]
    ax1.legend(custom_lines, ['cluster 1', 'cluster 2'], loc='lower left')

    ax1.text(-0.22, 1.02, "A",
             fontsize=10,
             transform=ax1.transAxes)

    ax2.text(-0.22, 1.02, "B",
             fontsize=10,
             transform=ax2.transAxes)

    # fig.tight_layout(pad=1, h_pad=0, w_pad=1)

    ax3 = plt.axes((0.25, 0.7, 0.2, 0.2))
    barcols = [clust_cols[i] for i in clusts]
    ax3.bar(range(len(test_loss)), test_loss, color=barcols)
    ax3.plot(ind_best_model, test_loss[ind_best_model] + 0.01, 'k*', ms=3)
    ax3.set_yticklabels([0, 0.05, 0.1])
    ax3.set_xticklabels([])
    ax3.set_xticks([])
    ax3.set_xlabel('models')
    ax3.yaxis.set_label_position("right")
    ax3.set_ylabel('test loss')

    # fig.savefig("./resfigure/learnforcast.eps", dpi=600)
    # plt.show()
    print('[OK]')
    return fig

def pred_rec(rdf):
    rs = rdf.values
    rsq = rs ** 2

    p = np.polyfit(rs[:, 0] ** 2, rs[:, 1] ** 2, deg=1)
    x = np.arange(0.85, 1, 0.001)
    y = np.polyval(p, x)

    nclust = 2
    gm_model = GaussianMixture(n_components=nclust, random_state=0).fit(rsq[:, :1])
    gmres = gm_model.predict(rsq[:, :1])
    gmeans = gm_model.means_
    gcovs = gm_model.covariances_
    # print(gmeans)
    # print(gcovs)

    meta_r = np.corrcoef(rsq[gmres == 1].T)
    # print(meta_r)

    # Plot the coeficient of determinations of prediction and reconstruction against each other
    # axin_xlim = [0.98, 0.999]
    # axin_ylim = [0.9, 0.99]

    fig, ax = plt.subplots(1, 1)
    # axin = ax.inset_axes([0.45, 0.2, 0.47, 0.47])
    # plt.plot(x, y)
    # plt.text(0.96, 0.9, r'$a={0:.3f}$'.format(p[0]))
    # plt.plot(rs[:, 0]**2, rs[:, 1]**2, '.', color='#F9A448', ms='5')

    cols = ['b', '#F9A448', 'red']
    means = []
    stdevs = []
    for i in range(nclust):
        v = rsq[gmres == i]
        ax.plot(v[:, 0], v[:, 1], '.', color=cols[i], ms='5', label='cluster {}'.format(i + 1))
        # axin.plot(v[:, 0], v[:, 1], '.', color=cols[i], ms='8')
        means.append(v.mean(axis=0))
        stdevs.append(v.std(axis=0))

    means = np.array(means)
    stdevs = np.array(stdevs)

    # print('means', means)
    # print('stdevs', stdevs)

    # m = np.array([[axin_xlim[0], means[1, 0], means[1, 0]], [means[1, 1], means[1, 1], axin_ylim[0]]]).T

    meancol = 'gray'
    # axin.plot(axin_xlim[0], means[1, 1], 'o', clip_on=False, color=meancol)
    # axin.plot(means[1, 0], means[1, 1], 'x', color=meancol)
    # axin.plot(means[1, 0], axin_ylim[0], 'o', clip_on=False, color=meancol)

    # axin.plot(m[:, 0], m[:, 1], '--', lw=0.5, color=meancol)

    # axin.text(0.7, 0.85, r'$\rho={:.3f}$'.format(meta_r[0, 1]), transform=axin.transAxes)
    # axin.text(0.65, -0.15, r'$\mu_p={:.3f}$'.format(means[1, 0]),
    #           transform=axin.transAxes, color=meancol)
    # axin.text(-0.05, 0.7, r'$\mu_r={:.3f}$'.format(means[1, 1]),
    #           transform=axin.transAxes, horizontalalignment='right', color=meancol)
    # axin.text(0.7, 0.5, r'$\sigma_r={:.2f}$'.format(stdevs[1, 1]), transform=axin.transAxes)

    # tx = np.arange(axin_xlim[0] + 0.002, axin_xlim[1] - 0.001, 1e-4)
    # px = norm.pdf(tx, loc=means[1, 0], scale=stdevs[1][0])
    # axin.plot(tx, axin_ylim[0] + np.diff(axin_ylim)*0.03 + 0.7*1e-4* px, '-', clip_on=False, color='orange')
    # axin.plot(tx, axin_ylim[0] + 1e-4 * px, '-', clip_on=False, color='orange')

    # ty = np.arange(axin_ylim[0] + 0.033, axin_ylim[1] - 0.01, 1e-3)
    # py = norm.pdf(ty, loc=means[1, 1], scale=stdevs[1][1])
    # axin.plot(axin_xlim[0] + np.diff(axin_xlim)*0.03 + 0.7*1e-4* py, ty, '-', clip_on=False, color='orange')
    # axin.plot(axin_xlim[0] + 0.7 * 1e-4 * py, ty, '-', clip_on=False, color='orange')

    # ft, axt = plt.subplots(1,1)
    # axt.plot(ty, py)
    # axt.plot(means[1,1], 0, 'o')
    # plt.show()
    # exit()

    ax.legend()
    ax.set_xlabel(r"$r_\mathrm{prediction}^2$")
    ax.set_ylabel(r"$r_\mathrm{reconstruction}^2$")

    # print(p)

    # ax.set_ylim(-.1, 1)
    # ax.set_xlim(0.8, 1)
    # axin.set_xlim(axin_xlim)
    # axin.set_ylim(axin_ylim)
    # ax.indicate_inset_zoom(axin, edgecolor="black")
    # axin.set_xticks([])
    # axin.set_yticks([])

    # plt.savefig("./resfigure/mapper-maco.eps")

    # plt.show()
    print('[OK]')
    return fig

def reconstruction(df):
    # x_pred = df['x_pred'].values
    # x_test = df['x_valid'].values
    # Y_test = df[["Y_1_valid", "Y_2_valid"]].values
    cc_pred = df['cc_pred'].values
    cc_val = df['cc_valid'].values

    # compute correlation
    rec_perform = np.corrcoef(cc_val[:], cc_pred[:])
    # print(rec_perform)

    cp = scale(cc_pred)
    c = scale(cc_val)

    T = 59

    fig2, axs2 = plt.subplots(1, 2)

    axs2[0].plot(minmax_scale(c[:T]), label="original", color='k', alpha=1, lw=2)
    axs2[0].plot(minmax_scale(np.sign(rec_perform[0, 1]) * cp[:T]), label="reconstructed",
                 linestyle='-', color='#9EE004', alpha=1, lw=1.5)

    axs2[1].plot(scale(cc_val[:]), scale(np.sign(rec_perform[0, 1]) * cc_pred[:]), '.', alpha=0.2, color="#9EE004")
    axs2[1].plot([-1.9, 1.5], [-1.9, 1.5], 'k--')
    axs2[1].text(.05, .9, r'$r^2={:.2f}$'.format(rec_perform[0, 1] ** 2), transform=axs2[1].transAxes)

    axs2[0].set_xlabel(r'$t$ (simulation step)')
    axs2[0].set_ylabel(r'$z$')

    axs2[0].legend(loc="lower left")

    axs2[1].set_xlabel(r'normalized $z(t-1)$')
    axs2[1].set_ylabel(r'normalized $\hat{z}(t-1)$')
    axs2[1].set_xlim([-1.9, 1.5])
    axs2[1].set_ylim([-1.9, 1.5])
    # axs2[1].set_ylim(axs2[1].get_xlim())
    # axs2[1].set_xlim(axs2[1].get_ylim())

    axs2[0].text(-0.22, 1.02, "A",
                 fontsize=10,
                 transform=axs2[0].transAxes)

    axs2[1].text(-0.22, 1.02, "B",
                 fontsize=10,
                 transform=axs2[1].transAxes)

    # fig2.tight_layout(pad=1, h_pad=0, w_pad=1)

    # fig2.savefig("./resfigure/reconstruction.eps")
    # fig2.savefig("./resfigure/reconstruction.png")
    # plt.show()
    print('[OK]')
    return fig2