import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from scripts.config_runall import figures_root
from scripts.experiments.config_loader import get_config, resolve_paths
from scripts.experiments.experiment_registry import get_family_spec

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _get_default_paths(config_path=None):
    if config_path is not None:
        cfg = resolve_paths(get_config('tentmaps', config_path), _REPO_ROOT)
        paths = cfg.get('paths', {})
        final_res_path = paths['final_res_path']
        figure_path = paths.get('figure_path', figures_root)
        return final_res_path, figure_path

    from scripts.config_runall import CONFIG_TENTMAPS

    paths = CONFIG_TENTMAPS.get('paths', {})
    if 'final_res_path' not in paths:
        raise KeyError("CONFIG_TENTMAPS['paths']['final_res_path'] is required")

    final_res_path = Path(paths['final_res_path'])
    figure_path = Path(paths.get('figure_path', figures_root))
    return final_res_path, figure_path


def plot_tentmap_comparison(final_res_path, figure_path):
    final_res_path = Path(final_res_path)
    figure_path = Path(figure_path)

    family_spec = get_family_spec('tentmaps')

    # Create dataframe
    df = pd.read_csv(final_res_path / family_spec.combined_csv, index_col=0)

    # Sort by median values in ascending order
    grouped = df[['method', 'r']].groupby('method')
    df2 = pd.DataFrame({col:vals['r'] for col,vals in grouped},)
    meds = df2.median().sort_values(ascending=True, inplace=False)
    df2 = df2[meds.index]
    print(meds)

    # Plot
    fs = 20
    ticksize = 16

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(df2, color="tab:orange", ax=ax)
    sns.swarmplot(data=df2, color=".25", size=3, ax=ax)

    ax.set_ylim(-0.05, 1.05)
    ax.grid(True)

    ax.set_ylabel('Coef. of Determination', size=fs)
    ax.set_xlabel('Method', size=fs)
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=ticksize)
    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels([r'{:.1f}'.format(i) for i in ax.get_yticks()], fontsize=ticksize)

    plt.tight_layout()
    figure_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure_path / 'comparison_tentmap.png', dpi=300)
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

    return plot_tentmap_comparison(final_res_path=final_res_path,
                                   figure_path=figure_path)


if __name__ == '__main__':
    main()
