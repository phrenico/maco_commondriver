import argparse
from pathlib import Path

import pandas as pd
from scripts.experiments.lorenz_hypertune.htune_config import final_savefig_path
from scripts.experiments.lorenz_hypertune.htune_config import plot_htune
from scripts.experiments.config_loader import get_config, resolve_paths
import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parents[3]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None)
    args = parser.parse_args()
    cfg = resolve_paths(get_config('lorenz_htune', args.config), _REPO_ROOT)

    interim_save_path = cfg['paths']['interim_res_path']
    final_save_path = cfg['paths']['final_res_path']

    pca_df = pd.read_csv(interim_save_path / 'pca_htune.csv', index_col=0)
    ica_df = pd.read_csv(interim_save_path / 'ica_htune.csv', index_col=0)
    dca_df = pd.read_csv(interim_save_path / 'dca_htune.csv', index_col=0)
    sfa_df = pd.read_csv(interim_save_path / 'sfa_htune.csv', index_col=0)

    df = pd.concat([pca_df, ica_df, dca_df, sfa_df], ignore_index=True, axis=0)
    df.to_csv(final_save_path / 'htune.csv')

    fig, ax = plt.subplots(2, 4, figsize=(20, 10), sharex=True, sharey='row')
    plot_htune(pca_df, 'PCA', fig_axes=[fig, ax[0, 0], ax[1, 0]],
               yaxlabel=True, save=False)
    plot_htune(ica_df, 'ICA', fig_axes=[fig, ax[0, 1], ax[1, 1]],
               yaxlabel=False, save=False)
    plot_htune(dca_df, 'DCA', fig_axes=[fig, ax[0, 2], ax[1, 2]],
               yaxlabel=False, save=False)
    plot_htune(sfa_df, 'SFA', fig_axes=[fig, ax[0, 3], ax[1, 3]],
               yaxlabel=False, save=False)

    # Add A, B, C, D, E, F , G, H labels to subplots
    for i, label in enumerate(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']):
        ax[i//4, i%4].text(-0.1, 1.1, label, transform=ax[i//4, i%4].transAxes,
                           fontsize=20, va='top', ha='right')
    fig.tight_layout()

    # draw stars to selected coordinates
    star_kwargs = dict(s='*', color='r', fontsize=20, va='top', ha='center', weight='bold')
    ax[0, 0].text(4, 1, **star_kwargs)
    ax[0, 1].text(4, 1, **star_kwargs)
    ax[0, 2].text(4, 1, **star_kwargs)
    ax[0, 3].text(2, 1, **star_kwargs)

    fig.savefig(final_savefig_path / 'htune.png')