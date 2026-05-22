import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from scripts.config_runall import figures_root
from scripts.experiments.config_loader import get_config, resolve_paths
from scripts.plots.config_figgen import palette
from scripts.experiments.experiment_registry import get_family_spec

_REPO_ROOT = Path(__file__).resolve().parents[3]
family_spec = get_family_spec('lorenz')


def _get_default_paths(config_path=None):
    if config_path is not None:
        cfg = resolve_paths(get_config('lorenz', config_path), _REPO_ROOT)
        paths = cfg.get('paths', {})
        final_res_path = paths['final_res_path']
        figure_path = paths.get('figure_path', figures_root)
        return final_res_path, figure_path

    from scripts.config_runall import CONFIG_LORENZ

    paths = CONFIG_LORENZ.get('paths', {})
    if 'final_res_path' not in paths:
        raise KeyError("CONFIG_LORENZ['paths']['final_res_path'] is required")

    final_res_path = Path(paths['final_res_path'])
    figure_path = Path(paths.get('figure_path', figures_root))
    return final_res_path, figure_path


def plot_lorenz_comparison(final_res_path, figure_path):
    final_res_path = Path(final_res_path)
    figure_path = Path(figure_path)

    print('load from: ', final_res_path)
    print('save to: ', figure_path)

    df = pd.read_csv(final_res_path / family_spec.combined_csv)
    print(df.columns)

    method_order = df[['method', 'r']].groupby('method').median().sort_values(by='r',
                                                                      ascending=True).index

    fs = 20
    ticksize = 16

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x='method', y='r', hue='method',
                palette=palette, order=method_order, ax=ax)
    sns.swarmplot(data=df, x='method', y='r',
                  color='.25', alpha=0.5, size=4,
                  order=method_order, ax=ax)

    ax.set_ylim(-0.05, 1.05)
    ax.grid(True)

    ax.set_ylabel('Coef. of Determination', size=fs)
    ax.set_xlabel('Method', size=fs)
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=ticksize)
    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels([r'{:.1f}'.format(i) for i in ax.get_yticks()], fontsize=ticksize)

    figure_path.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(figure_path / 'comparisons_lorenz.png', dpi=300)
    return fig


def main(args=None, config_path=None, final_res_path=None, figure_path=None):
    if final_res_path is None or figure_path is None:
        if config_path is None:
            parser = argparse.ArgumentParser()
            parser.add_argument('--config', default='scripts/config_runall.py')
            parsed = parser.parse_args(args)
            config_path = parsed.config

        default_final_res_path, default_figure_path = _get_default_paths(config_path=config_path)
        if final_res_path is None:
            final_res_path = default_final_res_path
        if figure_path is None:
            figure_path = default_figure_path

    return plot_lorenz_comparison(final_res_path=final_res_path,
                                  figure_path=figure_path)


if __name__ == '__main__':
    main()