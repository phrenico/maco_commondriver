import numpy as np
import pandas

r = np.genfromtxt('./50SOMCors.txt')
df = pandas.DataFrame({'data_id': range(len(r)),
                       'r': r,
                       'method': len(r) * ['SomASOM'],
                       'dataset': len(r) * ['SomaLogmap']})
print(df.head())
df.to_csv('../SOM_res.csv')
