'''Script to run ShRec on tent map data-set'''
import argparse
from pathlib import Path

from tqdm import tqdm
from shrec.models import RecurrenceManifold

from cdriver.preprocessing.splitters import train_valid_test_split
from cdriver.savers.saver import save_results
from cdriver.evaluate.evalz import comp_ccorr, get_maxes
from cdriver.datagen.tent_map import gen_tentmapdata
from scripts.experiments.config_loader import get_config, resolve_paths

_REPO_ROOT = Path(__file__).resolve().parents[3]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None)
    args = parser.parse_args()
    cfg = resolve_paths(get_config('tentmaps', args.config), _REPO_ROOT)

    N = cfg['datagen']['N']
    dataset, params = gen_tentmapdata(cfg['datagen'])

    train_split = cfg['preprocessing']['train_split']
    valid_split = cfg['preprocessing']['valid_split']
    d_embed = cfg['methods']['shrec']['d_embed']

    maxcs = []
    for n_iter in tqdm(range(N)):
        data = dataset[n_iter]
        X = data[:, 1:]
        y = data[:, 0]

        X_train, _, z_train, X_valid, _valid, z_valid, X_test, __, z_test = train_valid_test_split(X, X, y, train_split, valid_split)

        model = RecurrenceManifold(d_embed=d_embed)
        y_recon = model.fit_predict(X_train)

        tau, c = comp_ccorr(z_train, y_recon)
        maxtau, maxc = get_maxes(tau, c)
        maxcs.append(maxc)

    df = save_results(fname=cfg['paths']['interim_res_path'] / 'shrec_res.csv',
                      r=maxcs, N=N, method='ShRec', dataset='tentmap')
