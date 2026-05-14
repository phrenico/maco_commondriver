'''Run Shrec experiments'''

from tqdm import tqdm

from cdriver.preprocessing.splitters import train_valid_test_split

from cdriver.savers.saver import save_results
from cdriver.evaluate.evalz import comp_ccorr, get_maxes
from cdriver.datagen.tent_map import gen_tentmapdata

from scripts.datagen_scripts.datagen_config import tentmapgen_params
from scripts.experiments.tentmaps.config_tentmapres import train_split, interim_res_path, valid_split
from shrec.models import RecurrenceManifold

# @title Fixed Coupling

N = tentmapgen_params['N']  # number of realizations
dataset, params = gen_tentmapdata(tentmapgen_params)

# Run the  Reconstructions on the Datasets

maxcs = []
for n_iter in tqdm(range(N)):
    data = dataset[n_iter]
    X = data[:, 1:]
    y = data[:, 0]

    X_train, _, z_train, X_valid, _valid, z_valid, X_test, __, z_test = train_valid_test_split(X, X, y, train_split,
                                                                                               valid_split)

    model = RecurrenceManifold(d_embed=3)

    y_recon = model.fit_predict(X_train)
    # model.fit(X_train)
    # y_recon = model.predict(X_test)

    tau, c = comp_ccorr(z_train, y_recon)
    maxtau, maxc = get_maxes(tau, c)
    maxcs.append(maxc)

# Save results
df = save_results(fname=interim_res_path / './shrec_res.csv',
                  r=maxcs,
                  N=N,
                  method='ShRec',
                  dataset='tentmap')


