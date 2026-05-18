import argparse
from pathlib import Path

from scripts.experiments.combine_utils import combine_result_files
from scripts.experiments.experiment_registry import get_family_spec
from scripts.experiments.config_loader import get_config, resolve_paths

_REPO_ROOT = Path(__file__).resolve().parents[3]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None)
    args = parser.parse_args()
    cfg = resolve_paths(get_config('logmaps', args.config), _REPO_ROOT)

    family_spec = get_family_spec('logmaps')
    interim_res_path = cfg['paths']['interim_res_path']
    final_res_path = cfg['paths']['final_res_path']

    df = combine_result_files(interim_res_path,
                              final_res_path / family_spec.combined_csv,
                              family_spec.result_files)
    print(df.head())