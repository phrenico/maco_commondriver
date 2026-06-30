"""Split the composite comparisons_res.png into three standalone subplots.

Reads the final result CSVs, applies the same palette and plot_sub styling as
plot_comparisons.py, and saves one figure per dataset family into
presentation/.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

_SCRIPT_PATH = Path(__file__).resolve()
_PRESENTATION_DIR = _SCRIPT_PATH.parents[1]
_REPO_ROOT = _SCRIPT_PATH.parents[2]

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.experiments.experiment_registry import get_plot_family_specs

_FINAL_DIR = _REPO_ROOT / 'paper_artifacts' / 'results' / 'final'
_OUT_DIR = _PRESENTATION_DIR

swarm_color = '.25'
swarm_size = 4
fs = 20
tick_size = 16

# Three families in plot order
FAMILY_KEYS = ('logmaps', 'tentmaps', 'lorenz')
OUTPUT_NAMES = ('comparisons_A_logistic_maps.png', 'comparisons_B_tent_maps.png', 'comparisons_C_lorenz_systems.png')

# Shared global palette built once from the most complete family (logmaps).
# Every method gets the same colour in every subplot.
SHARED_PALETTE = None


def build_palette(df):
    """Construct a method→colour palette sorted by median r (same as composite)."""
    medians = df[['method', 'r']].groupby('method').median().sort_values(
        by='r', ascending=True
    )
    methods = medians.index
    palette_cols = sns.color_palette('husl', len(methods))
    return dict(zip(methods, palette_cols))


def plot_sub_standalone(df, palette, title, out_path):
    """Single-family box+swarm plot matching the composite subpanel styling."""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    # Restrict palette to only the methods present in *this* dataframe.
    methods_present = df['method'].unique()
    local_palette = {m: palette[m] for m in methods_present if m in palette}
    palette = local_palette

    method_order = (
        df[['method', 'r']]
        .groupby('method')
        .median()
        .sort_values(by='r', ascending=True)
        .index
    )

    sns.boxplot(
        data=df, x='method', y='r', hue='method',
        palette=palette, order=method_order, ax=ax,
    )
    sns.swarmplot(
        data=df, x='method', y='r',
        color=swarm_color, alpha=0.5, size=swarm_size,
        order=method_order, ax=ax,
    )

    ax.set_title(title, fontsize=fs + 2, pad=12)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True)
    ax.set_ylabel('Coef. of Determination', size=fs)
    ax.set_xlabel('Method', size=fs)
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=90, horizontalalignment='center', fontsize=tick_size,
    )
    ax.set_yticklabels(
        [r'{:.1f}'.format(i) for i in ax.get_yticks()], fontsize=tick_size,
    )

    # Remove the redundant hue legend (method names are already on the x-axis)
    if ax.get_legend() is not None:
        ax.get_legend().remove()

    fig.tight_layout()
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(_OUT_DIR / out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {_OUT_DIR / out_path}')


def main():
    specs = get_plot_family_specs()  # logmaps, tentmaps, lorenz

    global SHARED_PALETTE

    # Build the shared palette once from the logmaps data (11 methods) and
    # reuse it for tentmaps (11 methods) and lorenz (10 methods, no AniSOM).
    logmap_csv = _FINAL_DIR / specs[0].combined_csv
    print(f'Building shared palette from {logmap_csv} …')
    df_logmap = pd.read_csv(logmap_csv)
    SHARED_PALETTE = build_palette(df_logmap)

    for family_key, spec, out_name in zip(FAMILY_KEYS, specs, OUTPUT_NAMES):
        if family_key == 'logmaps':
            df = df_logmap  # already loaded
        else:
            csv_path = _FINAL_DIR / spec.combined_csv
            print(f'Reading {csv_path} …')
            df = pd.read_csv(csv_path)

        title = spec.title
        plot_sub_standalone(df, SHARED_PALETTE, title, out_name)

    print(f'\nDone — {len(specs)} standalone plots in {_OUT_DIR}')


if __name__ == '__main__':
    main()
