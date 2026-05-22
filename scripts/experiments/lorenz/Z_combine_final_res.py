import argparse
from pathlib import Path

from scripts.experiments.combine_utils import combine_result_files
from scripts.experiments.experiment_registry import get_family_spec
from scripts.experiments.config_loader import get_config, resolve_paths

_REPO_ROOT = Path(__file__).resolve().parents[3]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='scripts/config_runall.py')
    args = parser.parse_args()
    cfg = resolve_paths(get_config('lorenz', args.config), _REPO_ROOT)

    family_spec = get_family_spec('lorenz')
    interim_res_path = cfg['paths']['interim_res_path']
    final_res_path = cfg['paths']['final_res_path']

    print('Starting to combine results from ', interim_res_path)
    print('Loading results from ', interim_res_path)

    print("Combining Lorenz's results")
    df = combine_result_files(interim_res_path,
                              final_res_path / family_spec.combined_csv,
                              family_spec.result_files)

    print("Saving combined results to ", final_res_path / family_spec.combined_csv)
    print('Saved combined results')