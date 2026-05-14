from scripts.config import example_logmap_final_res_path
from scripts.plots.example_logmap.plot_example_res import main as plot_example_logmap_res


REQUIRED_FILES = (
    'mappercoach_res.csv',
    'learning_curves.npy',
    'valid_loss.npy',
    'best_model.pth',
    'models.pkl',
    'r_values.csv',
)


def main():
    missing = [name for name in REQUIRED_FILES if not (example_logmap_final_res_path / name).exists()]
    if missing:
        missing_str = ', '.join(missing)
        raise FileNotFoundError(f'Missing example_logmap outputs: {missing_str}')
    print('example_logmap outputs are present.')
    plot_example_logmap_res()

if __name__ == '__main__':
    main()
