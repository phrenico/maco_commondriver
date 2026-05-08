
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scripts.plots.config_figgen import fig_path, lorenz_path, palette
from scripts.experiments.experiment_registry import get_family_spec

res_path = lorenz_path
family_spec = get_family_spec('lorenz')

print("load from: ", res_path)
print("save to: ", fig_path)



# Create dataframe
df = pd.read_csv(res_path / family_spec.combined_csv)
print(df.columns)

# Sort by median values in ascending order
method_order = df[['method', 'r']].groupby('method').median().sort_values(by='r',
                                                                  ascending=True).index


# Plot
fs = 20
ticksize = 16

fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=df, x='method', y='r', hue='method',
            palette=palette, order=method_order, ax=ax)
sns.swarmplot(data=df, x='method', y='r',
              color='.25', alpha=0.5, size=4,
              order=method_order, ax=ax)

ax.set_ylim(-0.05, 1.05)
ax.grid(True)


ax.set_ylabel('Coef. of Determination', size=fs)
ax.set_xlabel('Method', size=fs)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=ticksize)
ax.set_yticklabels([r'{:.1f}'.format(i) for i in ax.get_yticks()], fontsize=ticksize)


plt.tight_layout()
plt.savefig(fig_path/ 'misc' / 'comparisons_lorenz.png', dpi=300)

# plt.show()