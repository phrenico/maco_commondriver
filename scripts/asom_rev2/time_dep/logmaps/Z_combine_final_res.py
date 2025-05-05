import pandas as pd
import sys
sys.path.append('./')
sys.path.append('../../../')

from scripts.resgen.logmaps.config_logmapres import interim_res_path, final_res_path


pca_res = pd.read_csv(interim_res_path / 'pca_res.csv', index_col=0)
ica_res = pd.read_csv(interim_res_path / 'ica_res.csv', index_col=0)
cca_res = pd.read_csv(interim_res_path / 'cca_res.csv', index_col=0)
dcca_res = pd.read_csv(interim_res_path / 'dcca_res.csv', index_col=0)
gilpin_res = pd.read_csv(interim_res_path / 'shrec_res.csv', index_col=0)
sfa_res = pd.read_csv(interim_res_path / 'sfa_res.csv', index_col=0)
dca_res = pd.read_csv(interim_res_path / 'dca_res.csv', index_col=0)

random_res = pd.read_csv(interim_res_path / 'random_res.csv', index_col=0)


anisom_res = pd.read_csv(interim_res_path / 'anisom_res.csv', index_col=0)
maco_res = pd.read_csv(interim_res_path / 'maco_res.csv', index_col=0)

df = pd.concat([pca_res,
                ica_res,
                cca_res,
                dcca_res,
                gilpin_res,
                sfa_res,
                dca_res,
                random_res,
                maco_res,
                anisom_res],
               ignore_index=False)

df.to_csv(final_res_path / 'logmaps_res.csv')

print(df.head())