import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# set seaborn grid
sns.set(style="whitegrid")

def main():
    # load data
    df = pd.read_csv('unified_results.csv')
    # add lines to dataframe for Lorenz ASOM
    line = {'r': np.nan, 'method': 'ASOM', 'dataset': 'lorenz'}
    df = df.append(line, ignore_index=True)


    print(df.head())
    print(df.dataset.unique())

    # sort values by the method name
    df2 = df.sort_values(by='method')

    print(df2.head())

    filter1 = df2.dataset == 'logmap_fixed'
    filter2 = df2.dataset == 'tentmap'
    filter3 = df2.dataset == 'lorenz'

    meds = df2.groupby(['method', 'dataset'])['r'].median().reset_index()
    # print(meds)
    order = meds[meds.dataset == 'logmap_fixed'].sort_values(by='r', ascending=True, inplace=False).method.tolist()
    print(order)
    # exit()

    # order = ['Random', 'PCA', 'ICA', 'DCA', 'CCA', 'DCCA', 'SFA',  'ASOM', 'ShRec', 'MaCo']
    common_kwargs = dict(x='method', y='r', order = order, palette=["tab:orange"])
    dot_kwargs = dict(color=".25", size=3, x='method', y='r', order = order)

    # plot boxplot
    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True, sharey=True)

    sns.boxplot(data=df2[filter1], ax=axs[0], **common_kwargs)
    sns.swarmplot(data=df2[filter1], **dot_kwargs, ax=axs[0])

    sns.boxplot(data=df2[filter2], ax=axs[1], **common_kwargs)
    sns.swarmplot(data=df2[filter2], **dot_kwargs, ax=axs[1])

    sns.boxplot(data=df2[filter3], ax=axs[2], **common_kwargs)
    sns.swarmplot(data=df2[filter3], **dot_kwargs, ax=axs[2])

    axs[0].set_xlabel('')
    axs[1].set_xlabel('')
    axs[2].set_xlabel('Method')

    axs[0].set_ylabel(r'$r^2$')
    axs[1].set_ylabel(r'$r^2$')
    axs[2].set_ylabel(r'$r^2$')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=0)

    # Add A B C labels
    axs[0].text(-0.07, 1.1, 'A', transform=axs[0].transAxes, fontsize=16, fontweight='bold', va='top')
    axs[1].text(-0.07, 1.1, 'B', transform=axs[1].transAxes, fontsize=16, fontweight='bold', va='top')
    axs[2].text(-0.07, 1.1, 'C', transform=axs[2].transAxes, fontsize=16, fontweight='bold', va='top')

    fig.savefig('./unified_results.png', dpi=300)
    # plt.show()


if __name__=="__main__":
    main()