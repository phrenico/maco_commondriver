from sklearn.decomposition import PCA, FastICA
from tqdm.auto import tqdm
import pandas as pd
from scripts.resgen.lorenzs.lorenz_hypertune.htune_config import create_htune_df, compute4all, plot_htune, interim_save_path, interim_savefig_path
import dca
DCA = dca.DynamicalComponentsAnalysis


ns_components = range(1, 7)
dfs = []
for n_components in tqdm(ns_components, desc='Components'):
    maxcs, amaxcs = compute4all(n_components, DCA)
    df = create_htune_df(maxcs, amaxcs, n_components, len(maxcs), 'DCA', 'lorenz')
    dfs.append(df)

df = pd.concat(dfs, ignore_index=False)
df.to_csv(interim_save_path/ 'dca_htune.csv')

f = plot_htune(df, 'DCA', save=True, path=interim_savefig_path)
# f.show()

