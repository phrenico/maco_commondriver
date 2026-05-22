import pandas as pd
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm

color_ICA = 'tab:orange'
color_PCA = 'tab:blue'
color_sfa = 'tab:green'
color_DCA = 'teal'

color_dict = dict(ICA=color_ICA, PCA=color_PCA, SFA=color_sfa, DCA=color_DCA)

def get_htune_paths(cfg):
    paths = {
        'interim_res_path': cfg['paths']['interim_res_path'],
        'final_res_path': cfg['paths']['final_res_path'],
        'figure_path': cfg['paths']['figure_path'],
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def get_data_path_template(cfg, repo_root):
    return str(Path(repo_root) / cfg['data']['data_path_template'])


def get_data(fname):
    data = np.load(fname)
    X = data['v'][:, 3:]
    z = data['v'][:, 1]
    return X, z


def compute4all(cfg, repo_root, n_components, method):
    from cdriver.evaluate.evalz import comp_ccorr, get_maxes
    from cdriver.preprocessing.splitters import train_valid_test_split

    N = cfg['data']['N']
    train_split = cfg['preprocessing']['train_split']
    valid_split = cfg['preprocessing']['valid_split']
    data_path_template = get_data_path_template(cfg, repo_root)
    maxcs = []
    amaxcs = []

    for n_iter in tqdm(range(N), desc='Iterations'):
        data_path = data_path_template.format(n_iter)
        X, z = get_data(data_path)

        (X_train, Y_train, z_train,
         X_valid, Y_valid, z_valid,
         X_test, Y_test, z_test) = train_valid_test_split(X, X, z, train_split, valid_split)


        if method.__name__ == "DynamicalComponentsAnalysis":
            model = method(d=n_components, T=5, n_init=10)
        else:
            model = method(n_components=n_components, random_state=0)
        model.fit(X_train)
        zpred = model.transform(X_valid)

        m = max([get_maxes(*comp_ccorr(z_valid, zpred[:, j]))[1] for j in range(n_components)])
        a = np.argmax([get_maxes(*comp_ccorr(z_valid, zpred[:, j]))[1] for j in range(n_components)]) + 1
        maxcs.append(m)
        amaxcs.append(a)
    return maxcs, amaxcs


def create_htune_df(coefs, wincomp, n_components, N, method, dataset):
    """Saves results to a file and returns the dataframe
    """
    data_dict = dict(coefs=coefs, wcomp=wincomp, n_components=n_components, method=method, dataset=dataset)
    df = pd.DataFrame(data_dict)
    return df


def plot_htune(df, method, fig_axes=None, yaxlabel=True, save=False, path=Path('./')):
    import seaborn as sns
    import matplotlib.pyplot as plt

    if fig_axes is None:
        fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    else:
        fig, ax1, ax2 = fig_axes

    sns.boxplot(x='n_components', y='coefs', data=df, color=color_dict[method], ax=ax1)
    sns.swarmplot(x='n_components', y='coefs', data=df, color=".25", size=5, ax=ax1)
    # sns.swarmplot(x='n_components', y='wcomp', data=df, color=".25", size=3, ax=ax2)
    sns.violinplot(x='n_components', y='wcomp', data=df, color=color_dict[method],
                   bw_method=0.1, ax=ax2, linewidth=0)

    fs = 20
    ticksize = 16

    ax1.set_ylim(-0.02, 1.02)
    if yaxlabel:
        ax1.set_ylabel('Correlation coefficient', size=fs)
        ax2.set_ylabel('Winner component', size=fs)
    else:
        ax1.set_ylabel(None)
        ax2.set_ylabel(None)
    ax1.set_xlabel(None)

    ax1.set_title('{}'.format(method), size=fs)

    ax2.set_ylim(0.5, 6.5)
    ax2.set_xticks(range(0, 7))
    # ax2.set_xticklabelsize(ticksize)
    ax2.set_xlabel('# of components', size=fs)

    ax1.grid(True)
    ax2.grid(True)
    fig.tight_layout()
    if save:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        plt.savefig(path / '{}_htune.png'.format(method))
    return fig
