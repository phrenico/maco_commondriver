import argparse
from pathlib import Path

from tqdm.auto import tqdm
import pandas as pd
from scripts.experiments.lorenz_hypertune.htune_config import create_htune_df, compute4all, interim_save_path
from scripts.experiments.config_loader import get_config, resolve_paths
import dca
DCA = dca.DynamicalComponentsAnalysis

_REPO_ROOT = Path(__file__).resolve().parents[3]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None)
    args = parser.parse_args()
    cfg = resolve_paths(get_config('lorenz_htune', args.config), _REPO_ROOT)

    ns_components = cfg['sweep']['ns_components']
    dfs = []
    for n_components in tqdm(ns_components, desc='Components'):
        maxcs, amaxcs = compute4all(n_components, DCA)
        df = create_htune_df(maxcs, amaxcs, n_components, len(maxcs), 'DCA', 'lorenz')
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=False)
    df.to_csv(interim_save_path / 'dca_htune.csv')

