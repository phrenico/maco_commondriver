
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

pca_res = pd.read_csv('./pca_res.csv', index_col=0)
ica_res = pd.read_csv('./ica_res.csv', index_col=0)
cca_res = pd.read_csv('./cca_res.csv', index_col=0)
dcca_res = pd.read_csv('./dcca_res.csv', index_col=0)
gilpin_res = pd.read_csv('./shrec_res.csv', index_col=0)
sfa_res = pd.read_csv('./sfa_res.csv', index_col=0)
dca_res = pd.read_csv('./dca_res.csv', index_col=0)

random_res = pd.read_csv('./random_res.csv', index_col=0)

som_res = pd.read_csv('./SOM_res.csv', index_col=0)


# Create dataframe
df = pd.concat([pca_res,
                ica_res,
                cca_res,
                dcca_res,
                gilpin_res,
                som_res,
                sfa_res,
                dca_res,
                random_res], ignore_index=False)



# Sort by median values in ascending order
grouped = df[['method', 'r']].groupby('method')
df2 = pd.DataFrame({col:vals['r'] for col,vals in grouped},)
meds = df2.median().sort_values(ascending=True, inplace=False)
df2 = df2[meds.index]
print(meds)

sns.boxplot(df2, color="r")
sns.swarmplot(data=df2, color=".25", size=3)

plt.grid(True)

plt.ylabel('Correlation coefficient')
plt.xlabel('Method')
plt.savefig('comparisons.png')

plt.show()