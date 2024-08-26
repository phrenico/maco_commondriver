"""Generates the Figure about prediction performance

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale
from sklearn.cluster import KMeans
from matplotlib.lines import Line2D

plt.style.use('./figure_twocol_config.mplstyle')

print("Generate Figure 6. (learnforcast.eps)")
# Load data
df = pd.read_csv('./resdata/mappercoach_res.csv')
x_pred = df['x_pred'].values
x_valid = df['x_valid'].values
Y_test= df[["Y_1_valid", "Y_2_valid"]].values
cc_pred= df['cc_pred'].values
cc_val = df['cc_valid'].values
learnings = np.load('./resdata/learning_curves.npy')
test_loss = np.load('./resdata/test_loss.npy')


# find best model
ind_best_model = np.argmin(test_loss)


# Compute correlations
r = np.corrcoef(x_valid, x_pred, rowvar=False)[0, 1]
# print(r**2)

# cluster the endpoints of learning curves into 2 clusters
nc = 2
km = KMeans(n_clusters=nc, random_state=2)
clusts = km.fit_predict(learnings[-1:, :].T)
# print(np.sum(clusts==0), np.sum(clusts==1))

#visualize reasults
fig, axs = plt.subplots(1, 2)

ax1 = axs[0]
ax2 = axs[1]

clust_cols = ['#F9A448', 'b']

for i in range(nc):
    _ = ax1.plot(learnings[:, i==clusts], color=clust_cols[i])

ax2.plot(minmax_scale(x_valid), minmax_scale(x_pred), '.', alpha=0.2, color='#F71616')
ax2.plot([0, 1], [0, 1], 'k--',)



ax1.set_xlabel('# epochs')
ax1.set_ylabel(R'$L$ (mean squared loss)')
ax1.set_yscale('log')
ax1.set_xscale('log')



ax2.text(0.05, .9, r'$r^2={:.3f}$'.format(r**2), transform=ax2.transAxes)
ax2.set_xlabel(r"$x(t)$")
ax2.set_ylabel(r"$\hat{x}(t)$")
ax2.set_xlim(0,1)
ax2.set_ylim(0,1)



custom_lines = [Line2D([0], [0], color=clust_cols[1], lw=2),
                Line2D([0], [0], color=clust_cols[0], lw=2)]
ax1.legend(custom_lines, ['cluster 1', 'cluster 2'], loc='lower left')

ax1.text(-0.22, 1.02, "A",
         fontsize=10,
         transform=ax1.transAxes)

ax2.text(-0.22, 1.02, "B",
         fontsize=10,
         transform=ax2.transAxes)


# fig.tight_layout(pad=1, h_pad=0, w_pad=1)

ax3 = plt.axes((0.25, 0.7, 0.2, 0.2))
barcols = [clust_cols[i] for i in clusts]
ax3.bar(range(len(test_loss)), test_loss, color=barcols)
ax3.plot(ind_best_model, test_loss[ind_best_model]+0.01, 'k*', ms=3)
ax3.set_yticklabels([0, 0.05, 0.1])
ax3.set_xticklabels([])
ax3.set_xticks([])
ax3.set_xlabel('models')
ax3.yaxis.set_label_position("right")
ax3.set_ylabel('test loss')

fig.savefig("./resfigure/learnforcast.eps", dpi=600)
# plt.show()
print('[OK]')