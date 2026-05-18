import argparse
from pathlib import Path

from scripts.experiments.config_loader import get_config, resolve_paths
from scripts.plots.example_logmap.plot_example_res import main as plot_example_logmap_res

_REPO_ROOT = Path(__file__).resolve().parents[3]

REQUIRED_FILES = (
    'mappercoach_res.csv',
    'learning_curves.npy',
    'valid_loss.npy',
    'best_model.pth',
    'models.pkl',
    'r_values.csv',
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None)
    args = parser.parse_args()
    cfg = resolve_paths(get_config('example_logmap', args.config), _REPO_ROOT)

    final_res_path = cfg['paths']['final_res_path']
    missing = [name for name in REQUIRED_FILES if not (final_res_path / name).exists()]
    if missing:
        missing_str = ', '.join(missing)
        raise FileNotFoundError(f'Missing example_logmap outputs: {missing_str}')
    print('example_logmap outputs are present.')
    plot_example_logmap_res()


if __name__ == '__main__':
    main()
