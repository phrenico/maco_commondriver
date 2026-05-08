from scripts.config import project_path
import pandas as pd
import seaborn as sns
from scripts.resgen.experiment_registry import get_family_spec

# Define paths
fig_path = project_path / 'figures'

noise_length_path = project_path / 'results/final/noise_length'
logmaps_path = project_path / 'results/final/'
lorenzs_path = project_path / 'results/final/'
tentmaps_path = project_path / 'results/final/'

# define colors
logmap_spec = get_family_spec('logmaps')

try:
    df = pd.read_csv(logmaps_path / logmap_spec.combined_csv)
    medians = df[['method', 'r']].groupby('method').median().sort_values(by='r', ascending=True)
    methods = medians.index
    palette_cols = sns.color_palette("husl", len(methods))
    palette = dict(zip(methods, palette_cols))

    box_color = palette['MaCo']
except:
    print('No logmaps_res.csv found, it can cause problems with the colormap...')
swarm_color = '.25'
swarm_size = 4

fs = 20
tick_size = 16