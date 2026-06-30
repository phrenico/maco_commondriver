"""Split example_logmap_res.png (2×3 composite) into six standalone subplots.

Replicates the exact plot functions from scripts/plots/example_logmap/plot_example_res.py,
rendering each subpanel as its own figure saved into presentation/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import minmax_scale, scale

_SCRIPT_PATH = Path(__file__).resolve()
_PRESENTATION_DIR = _SCRIPT_PATH.parents[1]
_REPO_ROOT = _SCRIPT_PATH.parents[2]
_RES_DIR = _REPO_ROOT / 'paper_artifacts' / 'results' / 'final' / 'example_logmap'
_OUT_DIR = _PRESENTATION_DIR

_OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Plot A — Learning curves (training loss over epochs, log-log, clustered)
# ---------------------------------------------------------------------------
def plot_A_learning_curves():
    learnings = np.load(_RES_DIR / 'learning_curves.npy')  # (epochs, models)
    nc = 2
    km = KMeans(n_clusters=nc, random_state=2)
    clusts = 1 - km.fit_predict(learnings[-1:, :].T)
    clust_cols = ['#F9A448', 'b']

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i in range(nc):
        ax.plot(learnings[:, i == clusts], color=clust_cols[i])
    ax.set_xlabel('# epochs')
    ax.set_ylabel(r'$L$ (mean squared loss)')
    ax.set_yscale('log')
    ax.set_xscale('log')
    custom_lines = [Line2D([0], [0], color=clust_cols[1], lw=2),
                    Line2D([0], [0], color=clust_cols[0], lw=2)]
    ax.legend(custom_lines, ['cluster 1', 'cluster 2'], loc='lower left')
    ax.set_title('A — Learning Curves (Prediction Task)', fontsize=14, pad=10)
    fig.tight_layout()
    fig.savefig(_OUT_DIR / 'example_A_learning_curves.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved A_learning_curves.png')


# ---------------------------------------------------------------------------
# Plot B — Validation loss bar chart
# ---------------------------------------------------------------------------
def plot_B_validation_loss():
    learnings = np.load(_RES_DIR / 'learning_curves.npy')
    valid_loss = np.load(_RES_DIR / 'valid_loss.npy')
    nc = 2
    km = KMeans(n_clusters=nc, random_state=2)
    clusts = 1 - km.fit_predict(learnings[-1:, :].T)
    clust_cols = ['#F9A448', 'b']
    ind_best_model = np.argmin(valid_loss)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    barcols = [clust_cols[i] for i in clusts]
    ax.bar(range(len(valid_loss)), valid_loss, color=barcols)
    ax.plot(ind_best_model, valid_loss[ind_best_model] + 0.01, 'k*', ms=10)
    ax.set_yscale('log')
    ax.set_xticks([])
    ax.set_xlabel('models')
    ax.set_ylabel('validation loss')
    ax.set_title('B — Validation Loss (10 Models)', fontsize=14, pad=10)
    fig.tight_layout()
    fig.savefig(_OUT_DIR / 'example_B_validation_loss.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved B_validation_loss.png')


# ---------------------------------------------------------------------------
# Plot C — Prediction scatter (normalized x_test vs x_pred)
# ---------------------------------------------------------------------------
def plot_C_prediction_scatter():
    df = pd.read_csv(_RES_DIR / 'mappercoach_res.csv')
    x_pred = df['x_pred'].values
    x_test = df['x_test'].values
    r = np.corrcoef(x_test, x_pred, rowvar=False)[0, 1]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(minmax_scale(x_test), minmax_scale(x_pred), '.', alpha=1., color='#F71616')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.text(0.05, .9, r'$r^2={:.3f}$'.format(r ** 2), transform=ax.transAxes, fontsize=14)
    ax.set_xlabel(r'$x(t)$')
    ax.set_ylabel(r'$\hat{x}(t)$')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(r'C — Prediction: $x(t)$ vs $\hat{x}(t)$', fontsize=14, pad=10)
    fig.tight_layout()
    fig.savefig(_OUT_DIR / 'example_C_prediction_scatter.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved C_prediction_scatter.png')


# ---------------------------------------------------------------------------
# Plot D — Time series reconstruction (original vs reconstructed z)
# ---------------------------------------------------------------------------
def plot_D_timeseries_reconstruction():
    df = pd.read_csv(_RES_DIR / 'mappercoach_res.csv')
    cc_pred = df['cc_pred'].values
    cc_val = df['cc_test'].values
    rec_perform = np.corrcoef(cc_val[:], cc_pred[:])
    cp = scale(cc_pred)
    c = scale(cc_val)
    T = 59

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(minmax_scale(c[:T]), label='original', color='k', alpha=1, lw=2)
    ax.plot(minmax_scale(np.sign(rec_perform[0, 1]) * cp[:T]), label='reconstructed',
            linestyle='-', color='#9EE004', alpha=1, lw=1.5)
    ax.set_xlabel(r'$t$ (simulation step)')
    ax.set_ylabel(r'$z$')
    ax.legend(loc='lower left')
    ax.set_title('D — Reconstructed Time Series (Best Model)', fontsize=14, pad=10)
    fig.tight_layout()
    fig.savefig(_OUT_DIR / 'example_D_timeseries_reconstruction.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved D_timeseries_reconstruction.png')


# ---------------------------------------------------------------------------
# Plot E — Reconstruction scatter (normalized z vs ẑ)
# ---------------------------------------------------------------------------
def plot_E_reconstruction_scatter():
    df = pd.read_csv(_RES_DIR / 'mappercoach_res.csv')
    cc_pred = df['cc_pred'].values
    cc_val = df['cc_test'].values
    rec_perform = np.corrcoef(cc_val[:], cc_pred[:])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(scale(cc_val[:]), scale(np.sign(rec_perform[0, 1]) * cc_pred[:]),
            '.', alpha=1., color='#9EE004')
    ax.plot([-1.9, 1.5], [-1.9, 1.5], 'k--')
    ax.text(.05, .9, r'$r^2={:.2f}$'.format(rec_perform[0, 1] ** 2),
            transform=ax.transAxes, fontsize=14)
    ax.set_xlabel(r'normalized $z(t-1)$')
    ax.set_ylabel(r'normalized $\hat{z}(t-1)$')
    ax.set_xlim([-1.9, 1.5])
    ax.set_ylim([-1.9, 1.5])
    ax.set_title(r'E — Reconstruction: $z(t-1)$ vs $\hat{z}(t-1)$', fontsize=14, pad=10)
    fig.tight_layout()
    fig.savefig(_OUT_DIR / 'example_E_reconstruction_scatter.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved E_reconstruction_scatter.png')


# ---------------------------------------------------------------------------
# Plot F — Correlation coefficients (r²_prediction vs r²_reconstruction)
# ---------------------------------------------------------------------------
def plot_F_correlation_coefficients():
    rdf = pd.read_csv(_RES_DIR / 'r_values.csv', index_col=0)
    rs = rdf.values  # [r_predict, r_reconst]
    rsq = rs ** 2
    nclust = 2
    gm_model = GaussianMixture(n_components=nclust, random_state=0).fit(rsq[:, :1])
    gmres = gm_model.predict(rsq[:, :1])

    cols = ['b', '#F9A448'][::-1]
    cluster_names = ['1', '2'][::-1]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i in range(nclust):
        v = rsq[gmres == i]
        ax.plot(v[:, 0], v[:, 1], '.', color=cols[i], ms=12,
                label='cluster {}'.format(cluster_names[i]))
    ax.set_xlabel(r'$r_\mathrm{prediction}^2$', fontsize=13)
    ax.set_ylabel(r'$r_\mathrm{reconstruction}^2$', fontsize=13)
    ax.set_ylim(-.1, 1)
    ax.set_xlim(0.65, 1)
    ax.legend(loc='lower right')
    ax.set_title(r'F — $r^2$: Prediction vs Reconstruction (10 Models)',
                 fontsize=14, pad=10)
    fig.tight_layout()
    fig.savefig(_OUT_DIR / 'example_F_correlation_coefficients.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved F_correlation_coefficients.png')


# ---------------------------------------------------------------------------
def main():
    print('Generating standalone example_logmap subplots …')
    plot_A_learning_curves()
    plot_B_validation_loss()
    plot_C_prediction_scatter()
    plot_D_timeseries_reconstruction()
    plot_E_reconstruction_scatter()
    plot_F_correlation_coefficients()
    print(f'\nDone — 6 standalone plots in {_OUT_DIR}')


if __name__ == '__main__':
    main()
