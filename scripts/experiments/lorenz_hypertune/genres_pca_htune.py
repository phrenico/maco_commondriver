import argparse
from pathlib import Path

from sklearn.decomposition import PCA
from tqdm import tqdm
import pandas as pd
from scripts.experiments.lorenz_hypertune.htune_config import create_htune_df, compute4all, get_htune_paths
from scripts.experiments.config_loader import get_config, resolve_paths

_REPO_ROOT = Path(__file__).resolve().parents[3]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None)
    args = parser.parse_args()
    cfg = resolve_paths(get_config('lorenz_htune', args.config), _REPO_ROOT)

    ns_components = cfg['sweep']['ns_components']
    paths = get_htune_paths(cfg)
    dfs = []
    for n_components in tqdm(ns_components):
        maxcs, amaxcs = compute4all(cfg, _REPO_ROOT, n_components, PCA)
        df = create_htune_df(maxcs, amaxcs, n_components, len(maxcs), 'PCA', 'lorenz')
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=False)
    df.to_csv(paths['interim_res_path'] / 'pca_htune.csv')

