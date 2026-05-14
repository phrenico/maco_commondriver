import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scripts.plots.config_figgen import box_color, swarm_color, fig_path, noise_length_path



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

    plt.savefig(save_path/ 'noise_length_res.png')
    return fig
    
def plot_all_nl():
    len_df = pd.read_csv(noise_length_path / './length_maco_res.csv')
    noise_df = pd.read_csv(noise_length_path / './noise_maco_res.csv')

    fig = plot_noise_length(noise_df, len_df, fig_path)
    return fig


def main():
    fig = plot_all_nl()
    # move_figure(fig, 0, 0)
    # plt.show()

  


if __name__ == '__main__':
    main()
    