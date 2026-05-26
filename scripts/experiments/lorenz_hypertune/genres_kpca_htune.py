from sklearn.decomposition import KernelPCA
from tqdm import tqdm
import pandas as pd
from scripts.experiments.lorenz_hypertune.htune_config import create_htune_df, compute4all, interim_save_path


n_components_range = range(1, 7)
dfs = []
for n_components in tqdm(n_components_range):
    maxcs, amaxcs = compute4all(n_components, KernelPCA)
    df = create_htune_df(maxcs, amaxcs, n_components, len(maxcs), 'kPCA', 'lorenz')
    dfs.append(df)

df = pd.concat(dfs, ignore_index=False)
df.to_csv(interim_save_path / 'kpca_htune.csv')
