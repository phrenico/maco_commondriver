import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm
from cdriver.network.maco import MaCo
from cdriver.savers.saver import save_results
from cdriver.datagen.logmap import gen_logmapdata
import torch
from scripts.experiments.config_loader import get_config, resolve_paths
from scripts.experiments.maco_utils import (build_series_loaders,
                                       get_default_device,
                                       score_latent_reconstruction,
                                       train_and_select_best_model)

_REPO_ROOT = Path(__file__).resolve().parents[3]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None)
    args = parser.parse_args()
    cfg = resolve_paths(get_config('logmaps', args.config), _REPO_ROOT)
