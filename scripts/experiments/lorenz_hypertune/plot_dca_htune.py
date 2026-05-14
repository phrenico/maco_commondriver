import pandas as pd
from scripts.experiments.lorenz_hypertune.htune_config import plot_htune

df = pd.read_csv('dca_htune.csv', index_col=0)

f = plot_htune(df, 'DCA', save=True)