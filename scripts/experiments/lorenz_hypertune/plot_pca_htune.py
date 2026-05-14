import pandas as pd
from scripts.experiments.lorenz_hypertune.htune_config import plot_htune

df = pd.read_csv('pca_htune.csv', index_col=0)

f = plot_htune(df, 'PCA', save=True)