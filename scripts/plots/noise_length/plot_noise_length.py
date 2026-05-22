import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from scripts.config_runall import figures_root
from scripts.experiments.config_loader import get_config, resolve_paths
from scripts.plots.config_figgen import box_color, swarm_color

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _get_default_paths(config_path=None):
    if config_path is not None:
        cfg = resolve_paths(get_config('noise_length', config_path), _REPO_ROOT)
        paths = cfg.get('paths', {})
        final_res_path = paths['final_res_path']
        figure_path = paths.get('figure_path', figures_root)
        return final_res_path, figure_path

    from scripts.config_runall import CONFIG_NOISE_LENGTH

    paths = CONFIG_NOISE_LENGTH.get('paths', {})
    if 'final_res_path' not in paths:
        raise KeyError("CONFIG_NOISE_LENGTH['paths']['final_res_path'] is required")

    final_res_path = Path(paths['final_res_path'])
    figure_path = Path(paths.get('figure_path', figures_root))
    return final_res_path, figure_path



def plot_sub(ax, df, xlabel, ylabel, intlabels=False):
    Ls = df.L.unique()

    sns.boxplot(x='L', y='r2', data=df, color=box_color, ax=ax)
    sns.swarmplot(x='L', y='r2', data=df, color=swarm_color, ax=ax)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)

    rotation = 45
    xticks = ax.get_xticks()
    if intlabels:
        ax.set_xticklabels(['{:.0f}'.format(L) for L in Ls],
                           rotation=rotation)
    else:
        # signal to noise ratio
        ax.set_xticklabels(['{:.1f}'.format(L) for L in (Ls / 0.283)*100 ],
                           rotation=rotation)

    ax.set_ylim(0, 1)


def plot_noise_length(df_noise, df_length, save_path):
    save_path = Path(save_path)

    len_ax_xlabel = r'Length of Time Series'
    noise_ax_xlabel = r'$\sigma_{\mathrm{noise}} / \sigma_{\mathrm{signal}} \times 100$ (%)'
    ylabel = r'$r^2$ Score'

    fig, axs = plt.subplots(1, 2, figsize=(10, 6), sharey=True)

    plot_sub(axs[0], df_length, len_ax_xlabel, ylabel, intlabels=True)
    plot_sub(axs[1], df_noise, noise_ax_xlabel, ylabel)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # write A and B on the subplots
    axs[0].text(-0.15, 1.1, 'A', transform=axs[0].transAxes,
                fontsize=16, fontweight='bold', va='top')
    axs[1].text(-0.15, 1.1, 'B', transform=axs[1].transAxes,
                fontsize=16, fontweight='bold', va='top')

    save_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path / 'noise_length_res.png')
    return fig
    
def plot_all_nl(res_path, figure_path):
    res_path = Path(res_path)
    figure_path = Path(figure_path)

    len_df = pd.read_csv(res_path / 'length_maco_res.csv')
    noise_df = pd.read_csv(res_path / 'noise_maco_res.csv')

    fig = plot_noise_length(noise_df, len_df, figure_path)
    return fig


def main(args=None, config_path=None, res_path=None, figure_path=None):
    if res_path is None or figure_path is None:
        if config_path is None:
            parser = argparse.ArgumentParser()
            parser.add_argument('--config', default=None)
            parsed = parser.parse_args(args)
            config_path = parsed.config

        default_res_path, default_figure_path = _get_default_paths(config_path=config_path)
        if res_path is None:
            res_path = default_res_path
        if figure_path is None:
            figure_path = default_figure_path

    fig = plot_all_nl(res_path=res_path, figure_path=figure_path)
    # move_figure(fig, 0, 0)
    # plt.show()

    return fig

  


if __name__ == '__main__':
    main()
    