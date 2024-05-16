import pandas as pd
import sys
sys.path.append('../')
from htune_common import  plot_htune

df = pd.read_csv('pca_htune.csv', index_col=0)

f = plot_htune(df, 'PCA', save=True)