import argparse
from pathlib import Path

from scripts.experiments.config_loader import get_config, resolve_paths
from scripts.plots.noise_length import plot_noise_length

_REPO_ROOT = Path(__file__).resolve().parents[3]

REQUIRED_FILES = (
    'length_maco_res.csv',
    'noise_maco_res.csv',
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None)
    args = parser.parse_args()
    cfg = resolve_paths(get_config('noise_length', args.config), _REPO_ROOT)
    noise_length_final_res_path = cfg['paths']['final_res_path']
    missing = [name for name in REQUIRED_FILES if not (noise_length_final_res_path / name).exists()]
    if missing:
        missing_str = ', '.join(missing)
        raise FileNotFoundError(f'Missing noise_length outputs: {missing_str}')
    print('noise_length outputs are present.')
    if hasattr(plot_noise_length, 'plot_all_nl'):
        plot_noise_length.plot_all_nl()
    elif hasattr(plot_noise_length, 'plot_all'):
        plot_noise_length.plot_all()
    elif hasattr(plot_noise_length, 'main'):
        plot_noise_length.main()
    else:
        raise AttributeError('No callable plotting entrypoint found in scripts.plots.noise_length.plot_noise_length')


if __name__ == '__main__':
    main()
