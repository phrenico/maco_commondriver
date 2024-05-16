import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from setuptools._distutils.msvccompiler import normalize_and_reduce_paths

# set seaborn grid
sns.set(style="whitegrid")

def main():
    # load data
    df = pd.read_csv('unified_results.csv')
    # add lines to dataframe for Lorenz ASOM
    line = {'r': np.nan, 'method': 'ASOM', 'dataset': 'lorenz'}
    df = df.append(line, ignore_index=True)

    meds = df.groupby(['method', 'dataset'])['r'].median().reset_index()
    iqr = df.groupby(['method', 'dataset'])['r'].quantile(0.75) - df.groupby(['method', 'dataset'])['r'].quantile(0.25)

    lm_filter = meds.dataset == 'logmap_fixed'
    sm = meds[lm_filter].sort_values(by='r', ascending=True, inplace=False)
    columns = sm['method'].tolist()
    print(sm)


    # transform the table as methods as columns and datasets as rows and 'r' as values
    df2 = meds.pivot(index='dataset', columns='method', values='r')

    # reorder rows [0, 2, 1]
    df2 = df2.reindex(['logmap_fixed', 'tentmap', 'lorenz'])
    # rename rows
    df2 = df2.rename(index={'logmap_fixed': 'Logistic map', 'tentmap': 'Tent map', 'lorenz': 'Lorenz system'})
    print(df2)



    print(df2[columns].style.format(precision=2, na_rep='-').highlight_max(axis=1, props='color:blue; font-weight:bold;')
.to_latex(convert_css=True))



if __name__=="__main__":
    main()