from sksfa import SFA
from tqdm import tqdm
import pandas as pd
from scripts.experiments.lorenz_hypertune.htune_config import create_htune_df, compute4all, interim_save_path



ns_components = range(1, 7)
dfs = []
for n_components in tqdm(ns_components):
    maxcs, amaxcs= compute4all(n_components, SFA)
    df = create_htune_df(maxcs, amaxcs, n_components, len(maxcs), 'SFA', 'lorenz')
    dfs.append(df)

df = pd.concat(dfs, ignore_index=False)
df.to_csv(interim_save_path / 'sfa_htune.csv')

