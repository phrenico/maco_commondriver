
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

pca_res = pd.read_csv('./pca_res.csv', index_col=0)
# ica_res = pd.read_csv('./ica_res.csv', index_col=0)
# cca_res = pd.read_csv('./cca_res.csv', index_col=0)
dcca_res = pd.read_csv('./dcca_res.csv', index_col=0)
# gilpin_res = pd.read_csv('./shrec_res.csv', index_col=0)
# sfa_res = pd.read_csv('./sfa_res.csv', index_col=0)
# dca_res = pd.read_csv('./dca_res.csv', index_col=0)

# random_res = pd.read_csv('./random_res.csv', index_col=0)

# som_res = pd.read_csv('./SOM_res.csv', index_col=0)
# anisom_res = pd.read_csv('./anisom_res.csv', index_col=0)
maco_res = pd.read_csv('./maco_res.csv', index_col=0)
# dummy_maco_res = pd.read_csv('./dummy_maco_res.csv', index_col=0)
# dummy_maco_res['method'] = "MaCo"

# Create dataframe
df = pd.concat([pca_res,
                # ica_res,
                # cca_res,
                dcca_res,
                # gilpin_res,
                # som_res,
                # sfa_res,
                # dca_res,
                # random_res,
                maco_res,
                # dummy_maco_res,
                # anisom_res
                ],
               ignore_index=False)



# Sort by median values in ascending order
grouped = df[['method', 'r']].groupby('method')
df2 = pd.DataFrame({col:vals['r'] for col,vals in grouped},)
meds = df2.median().sort_values(ascending=True, inplace=False)
df2 = df2[meds.index]
print(meds)


# Plot
fs = 20
ticksize = 16

fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(df2, color="tab:orange", ax=ax)
sns.swarmplot(data=df2, color=".25", size=3, ax=ax)

ax.grid(True)

ax.set_ylabel('Correlation coefficient', size=fs)
ax.set_xlabel('Method', size=fs)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=ticksize)
ax.set_yticklabels([r'{:.1f}'.format(i) for i in ax.get_yticks()], fontsize=ticksize)


plt.tight_layout()
plt.savefig('comparisons_lorenz.png', dpi=300)

# plt.show()