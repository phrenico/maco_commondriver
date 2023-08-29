import pandas as pd
from htune_common import plot_htune
import matplotlib.pyplot as plt


pca_df = pd.read_csv('./pca_tune/pca_htune.csv', index_col=0)
ica_df = pd.read_csv('./ica_tune/ica_htune.csv', index_col=0)
dca_df = pd.read_csv('./dca_tune/dca_htune.csv', index_col=0)
sfa_df = pd.read_csv('./sfa_tune/sfa_htune.csv', index_col=0)

fig, ax = plt.subplots(2, 4, figsize=(20, 10), sharex=True, sharey='row')
plot_htune(pca_df, 'PCA', fig_axes=[fig, ax[0, 0], ax[1, 0]],
           yaxlabel=True, save=False)
plot_htune(ica_df, 'ICA', fig_axes=[fig, ax[0, 1], ax[1, 1]],
           yaxlabel=False, save=False)
plot_htune(dca_df, 'DCA', fig_axes=[fig, ax[0, 2], ax[1, 2]],
           yaxlabel=False, save=False)
plot_htune(sfa_df, 'SFA', fig_axes=[fig, ax[0, 3], ax[1, 3]],
           yaxlabel=False, save=False)
fig.savefig('htune.png')