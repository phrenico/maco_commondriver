import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('./length_maco_res.csv')
Ls = df.L.unique()
print([int(i) for i in Ls])

plt.figure(figsize=(10, 6))
sns.boxplot(x='L', y='r2', data=df, color="tab:orange")
sns.swarmplot(x='L', y='r2', data=df, color=".25")

plt.xlabel(r'$n$ Length of Time Series')
plt.ylabel(r'$r^2$ Score')
plt.grid(True)

plt.xticks(plt.gca().get_xticks(), ['{:.0f}'.format(L) for L in Ls])

plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('length_maco_res.png')
plt.show()