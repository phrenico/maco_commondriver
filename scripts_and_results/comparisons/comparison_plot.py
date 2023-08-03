
import matplotlib.pyplot as plt
import pandas as pd

pca_res = pd.read_csv('./pca_res.csv', index_col=0)
ica_res = pd.read_csv('./ica_res.csv', index_col=0)
cca_res = pd.read_csv('./cca_res.csv', index_col=0)
dcca_res = pd.read_csv('./dcca_res.csv', index_col=0)
gilpin_res = pd.read_csv('./gilpin_res.csv', index_col=0)
som_res = pd.read_csv('./SOM_res.csv', index_col=0)

df = pd.concat([pca_res, ica_res, cca_res, dcca_res, gilpin_res, som_res], ignore_index=True)


# plt.figure()
# plt.hist(pca_res['r'], label='PCA')
# plt.hist(cca_res['r'], label='CCA')
# plt.hist(dcca_res['r'], label='DCCA')
# plt.legend()
# plt.show()

print(df[['method', 'r']].groupby('method').median())

df.boxplot(by='method', column='r')
plt.suptitle('')
plt.title('')
plt.ylabel('Correlation coefficient')
plt.xlabel('Method')
plt.savefig('comparisons.png')
plt.show()