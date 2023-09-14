import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('./noise_maco_res.csv')

Ls = df.L.unique()

print(df.L.unique())

plt.figure(figsize=(10, 6))
sns.boxplot(x='L', y='r2', data=df, color="tab:orange")
sns.swarmplot(x='L', y='r2', data=df, color=".25")

plt.xlabel(r'Noise $\sigma$')
plt.ylabel(r'$r^2$ score')
plt.grid(True)


plt.xticks(plt.gca().get_xticks(), ['{:.3f}'.format(L) for L in Ls])

plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('noise_maco_res.png')
plt.show()