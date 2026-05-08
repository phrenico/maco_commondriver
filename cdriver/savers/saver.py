import pandas as pd
import numpy as np


def save_results(fname, r, N, method, dataset, times=None):
    """Saves results to a file and returns the dataframe
    """
    if times is None:
        times = N * [np.nan]

    df = pd.DataFrame({
        'data_id': range(N),
        'r': r,
        'duration': times,
        'method': N * [method],
        'dataset': N * [dataset],
    })
    df.to_csv(fname, index=False)
    return df
