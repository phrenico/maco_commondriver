import matplotlib.pyplot as plt
import seaborn as sns
from config_figgen import palette, swarm_color, swarm_size, fs, tick_size, logmaps_path, lorenzs_path, tentmaps_path, fig_path
import pandas as pd

def plot_sub(ax, df, ax_kwargs={}):
    method_order = df[['method', 'r']].groupby('method').median().sort_values(by='r',
                                                                      ascending=True).index

    sns.boxplot(data=df, x='method', y='r', hue='method',
            palette=palette, order=method_order, ax=ax, legend=True)
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

def plot_comparisons(df_logmap, df_tentmap, df_lorenz, save_path=None):
    fig, axs = plt.subplots(1, 3, figsize=(12, 6), sharey=True)

    plot_sub(axs[0], df_logmap, ax_kwargs={'title': 'Logistic Maps'})
    plot_sub(axs[1], df_tentmap, ax_kwargs={'title': 'Tent Maps'})
    plot_sub(axs[2], df_lorenz, ax_kwargs={'title': 'Lorenz Systems'})

    axs[0].set_xlabel('')
    axs[2].set_xlabel('')
    handles, labels = axs[0].get_legend_handles_labels()
    axs[0].legend().remove()
    axs[1].legend().remove()
    axs[2].legend().remove()
    

    # reorder the labels and handles according to the order of the methods in palette
    labels_new = [i for i in palette.keys() if i in labels]
    labels_new.reverse()
    handles_new = [handles[labels.index(i)] for i in labels_new]

    axs[2].legend(handles_new, labels_new, loc='center left',
                  bbox_to_anchor=(1, 0.5) )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95], w_pad=0.5, pad=0.5)
    
    # print A B C on the subplots
    axs[0].text(-0.15, 1.1, 'A', transform=axs[0].transAxes,
                fontsize=16, fontweight='bold', va='top')
    axs[1].text(-0.05, 1.1, 'B', transform=axs[1].transAxes,
                fontsize=16, fontweight='bold', va='top')
    axs[2].text(-0.05, 1.1, 'C', transform=axs[2].transAxes,
                fontsize=16, fontweight='bold', va='top')
    

    if save_path:
        plt.savefig(save_path / 'comparisons_res.png')
    return fig


def main():
    #load data
    df_logmap = pd.read_csv(logmaps_path / 'logmaps_res.csv')
    df_tentmap = pd.read_csv(tentmaps_path / 'tentmaps_res.csv')
    df_lorenz = pd.read_csv(lorenzs_path / 'lorenzs_res.csv')

    fig = plot_comparisons(df_logmap, df_tentmap, df_lorenz, save_path=fig_path)
    # plt.show()

if __name__ == '__main__':
    main()