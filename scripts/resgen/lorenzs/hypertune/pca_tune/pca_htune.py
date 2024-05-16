from sklearn.decomposition import PCA, FastICA
from tqdm import tqdm
import pandas as pd
import sys
sys.path.append('../')
from htune_common import create_htune_df, compute4all, plot_htune



ns_components = range(1, 7)
dfs = []
for n_components in tqdm(ns_components):
    maxcs, amaxcs = compute4all(n_components, PCA)
    df = create_htune_df(maxcs, amaxcs, n_components, len(maxcs), 'PCA', 'lorenz')
    dfs.append(df)

df = pd.concat(dfs, ignore_index=False)
df.to_csv('pca_htune.csv')

f = plot_htune(df, 'PCA', save=True)

