'''Create unified result dataframe from all the results

1. load all the results for all the examples
2. create a unified dataframe
3. save the dataframe to a csv file
'''
import pandas as pd
import os
from pathlib import Path

# Load all the results
save_path = Path('/home/phrenico/Projects/Codes/maco_commondriver/scripts_and_results/comparisons/unified_results')
examples = ['logmap', 'tentmap', 'lorenz']

df_list = []
for folder in examples:
    rpath = Path(f'/home/phrenico/Projects/Codes/maco_commondriver/scripts_and_results/comparisons') / folder
    res_files = [i for i in os.listdir(rpath) if i.endswith('.csv')]
    print(folder)
    print([i for i in res_files if 'sfa' in i]) 

    df_list += [pd.read_csv(rpath / i, index_col=0) for i in res_files]
    print(df_list[-2].head())

df = pd.concat(df_list, ignore_index=True)
# print(df.head())
# print(len(df_list))

print(df[df.dataset=='tentmap'].method.unique())
df.to_csv(save_path / 'unified_results.csv')

