import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from scripts.experiments.config_loader import get_config, resolve_paths
from scripts.experiments.experiment_registry import get_plot_family_specs

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REQUIRED_PATH_KEYS = (
    'logmaps_final_res_path',
    'tentmaps_final_res_path',
    'lorenz_final_res_path',
    'figure_path',
)

swarm_color = '.25'
swarm_size = 4
fs = 20
tick_size = 16


def _get_comparison_plot_paths(config_path):
    if config_path is None:
        raise ValueError(
            'Shared comparison plot generation requires --config with '
            'CONFIG_COMPARISON_PLOTS.'
        )

    cfg = resolve_paths(get_config('comparison_plots', config_path), _REPO_ROOT)
    paths = cfg.get('paths', {})

    missing_keys = [key for key in _REQUIRED_PATH_KEYS if key not in paths]
    if missing_keys:
        missing_str = ', '.join(missing_keys)
        raise KeyError(
            'CONFIG_COMPARISON_PLOTS.paths is missing required keys: '
            f'{missing_str}'
        )

    return {key: Path(paths[key]) for key in _REQUIRED_PATH_KEYS}


def _build_palette(df_logmap):
    medians = df_logmap[['method', 'r']].groupby('method').median().sort_values(by='r', ascending=True)
    methods = medians.index
    palette_cols = sns.color_palette('husl', len(methods))
    return dict(zip(methods, palette_cols))


def plot_sub(ax, df, palette, ax_kwargs=None):
    if ax_kwargs is None:
        ax_kwargs = {}

    method_order = df[['method', 'r']].groupby('method').median().sort_values(by='r',
                                                                      ascending=True).index

    sns.boxplot(data=df, x='method', y='r', hue='method',
            palette=palette, order=method_order, ax=ax)
    sns.swarmplot(data=df, x='method', y='r',
                color=swarm_color, alpha=0.5, size=swarm_size,
                order=method_order, ax=ax)
    ax.set(**ax_kwargs)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True)

    ax.set_ylabel('Coef. of Determination', size=fs)
    ax.set_xlabel('Method', size=fs)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, horizontalalignment='center', fontsize=tick_size)
    ax.set_yticklabels([r'{:.1f}'.format(i) for i in ax.get_yticks()], fontsize=tick_size)


def plot_comparisons(df_logmap, df_tentmap, df_lorenz, figure_path):
    logmap_spec, tentmap_spec, lorenz_spec = get_plot_family_specs()
    palette = _build_palette(df_logmap)
    figure_path = Path(figure_path)

    fig, axs = plt.subplots(1, 3, figsize=(12, 6), sharey=True)

    plot_sub(axs[0], df_logmap, palette=palette, ax_kwargs={'title': logmap_spec.title})
    plot_sub(axs[1], df_tentmap, palette=palette, ax_kwargs={'title': tentmap_spec.title})
    plot_sub(axs[2], df_lorenz, palette=palette, ax_kwargs={'title': lorenz_spec.title})

    axs[0].set_xlabel('')
    axs[2].set_xlabel('')
    handles, labels = axs[0].get_legend_handles_labels()
    axs[0].legend().remove()
    axs[1].legend().remove()
    axs[2].legend().remove()

    labels_new = [name for name in palette.keys() if name in labels]
    labels_new.reverse()
    handles_new = [handles[labels.index(name)] for name in labels_new]

    axs[2].legend(handles_new, labels_new, loc='center left',
                  bbox_to_anchor=(1, 0.5))
    fig.tight_layout(rect=[0, 0.03, 1, 0.95], w_pad=0.5, pad=0.5)

    axs[0].text(-0.15, 1.1, 'A', transform=axs[0].transAxes,
                fontsize=16, fontweight='bold', va='top')
    axs[1].text(-0.05, 1.1, 'B', transform=axs[1].transAxes,
                fontsize=16, fontweight='bold', va='top')
    axs[2].text(-0.05, 1.1, 'C', transform=axs[2].transAxes,
                fontsize=16, fontweight='bold', va='top')

    figure_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure_path / 'comparisons_res.png')
    return fig


def main(args=None, config_path=None):
    if config_path is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', default=None)
        parsed = parser.parse_args(args)
        config_path = parsed.config

    paths = _get_comparison_plot_paths(config_path)
    logmap_spec, tentmap_spec, lorenz_spec = get_plot_family_specs()

    df_logmap = pd.read_csv(paths['logmaps_final_res_path'] / logmap_spec.combined_csv)
    df_tentmap = pd.read_csv(paths['tentmaps_final_res_path'] / tentmap_spec.combined_csv)
    df_lorenz = pd.read_csv(paths['lorenz_final_res_path'] / lorenz_spec.combined_csv)

    return plot_comparisons(df_logmap, df_tentmap, df_lorenz, figure_path=paths['figure_path'])


if __name__ == '__main__':
    main()