import numpy as np
import pandas

r = np.genfromtxt('./50SOMCors.txt')
df = pandas.DataFrame({'data_id': range(len(r)),
                       'r': r,
                       # 'r': np.sqrt(r),  # if it was r**2
                       'method': len(r) * ['SOM'],
                       'dataset': len(r) * ['SomaLogmap']})
print(df.head())
df.to_csv('../SOM_res.csv')
