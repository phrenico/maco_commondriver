
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scripts.config_runall import figures_root as figure_path, tentmaps_final_res_path as final_res_path
from scripts.experiments.experiment_registry import get_family_spec


def main():
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
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=ticksize)
    ax.set_yticklabels([r'{:.1f}'.format(i) for i in ax.get_yticks()], fontsize=ticksize)

    plt.tight_layout()
    figure_path.mkdir(parents=True, exist_ok=True)
    (figure_path / 'misc').mkdir(parents=True, exist_ok=True)
    plt.savefig(figure_path / 'misc' / 'comparison_tentmap.png', dpi=300)


if __name__ == '__main__':
    main()
