import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale, minmax_scale


# Load data
df = pd.read_csv('./resdata/mappercoach_res.csv')
x_pred = df['x_pred'].values
x_valid = df['x_valid'].values
Y_test= df[["Y_1_valid", "Y_2_valid"]].values
cc_pred= df['cc_pred'].values
cc_val = df['cc_valid'].values
learnings = np.load('./resdata/learning_curves.npy')


# Compute correlations
r = np.corrcoef(x_valid, x_pred, rowvar=False)[0, 1]
print(r**2)

#visualize reasults
fig, axs = plt.subplots(1, 2)
ax1 = axs[0]
ax2 = axs[1]

ax1.plot(np.array(learnings))
ax2.plot(minmax_scale(x_valid), minmax_scale(x_pred), 'b.', alpha=0.2)
ax2.plot([0, 1], [0, 1], 'k--',)


ax1.set_xlabel('# epochs')
ax1.set_ylabel(R'$L$ (mean squared loss)')
ax1.set_yscale('log')

ax2.text(0.05, .9, r'$r^2={:.3f}$'.format(r**2))
ax2.set_xlabel(r"$x(t)$")
ax2.set_ylabel(r"$\hat{x}(t)$")
ax2.set_xlim(0,1)
ax2.set_ylim(0,1)


ax1.text(-0.22, 1.02, "A",
         fontsize=16,
         transform=ax1.transAxes)

ax2.text(-0.22, 1.02, "B",
         fontsize=16,
         transform=ax2.transAxes)


fig.tight_layout(pad=1, h_pad=0, w_pad=1)

fig.savefig("./resdata/learnforcast.pdf", dpi=300)
# plt.show()