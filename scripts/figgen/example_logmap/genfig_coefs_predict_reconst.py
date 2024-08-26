import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylab as pl
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.stats import rankdata
from scipy.stats import norm
plt.style.use('./figure_onecol_config.mplstyle')



print("Generate Figure 8. (mapper-maco.eps)")
# Load Data
rdf = pd.read_csv('./resdata/r_values.csv', index_col=0)
rs = rdf.values
rsq = rs**2

p = np.polyfit(rs[:, 0]**2, rs[:, 1]**2, deg=1)
x = np.arange(0.85, 1, 0.001)
y = np.polyval(p, x)



nclust = 2
gm_model = GaussianMixture(n_components=nclust, random_state=0).fit(rsq[:, :1])
gmres = gm_model.predict(rsq[:, :1])
gmeans = gm_model.means_
gcovs = gm_model.covariances_
# print(gmeans)
# print(gcovs)

meta_r = np.corrcoef(rsq[gmres==1].T)
# print(meta_r)


# Plot the coeficient of determinations of prediction and reconstruction against each other
axin_xlim = [0.98, 0.999]
axin_ylim = [0.9, 0.99]

fig, ax = plt.subplots(1, 1)
axin = ax.inset_axes([0.45, 0.2, 0.47, 0.47])
# plt.plot(x, y)
# plt.text(0.96, 0.9, r'$a={0:.3f}$'.format(p[0]))
# plt.plot(rs[:, 0]**2, rs[:, 1]**2, '.', color='#F9A448', ms='5')

cols = ['b', '#F9A448', 'red']
means = []
stdevs = []
for i in range(nclust):
    v = rsq[gmres==i]
    ax.plot(v[:, 0], v[:, 1], '.', color=cols[i], ms='5', label='cluster {}'.format(i+1))
    axin.plot(v[:, 0], v[:, 1], '.', color=cols[i], ms='8')
    means.append(v.mean(axis=0))
    stdevs.append(v.std(axis=0))

means = np.array(means)
stdevs = np.array(stdevs)

# print('means', means)
# print('stdevs', stdevs)

m = np.array([[axin_xlim[0], means[1, 0], means[1, 0]], [means[1, 1], means[1, 1], axin_ylim[0]]]).T

meancol='gray'
axin.plot(axin_xlim[0], means[1, 1],  'o', clip_on=False, color=meancol)
axin.plot(means[1, 0], means[1, 1], 'x', color=meancol)
axin.plot(means[1, 0], axin_ylim[0],  'o', clip_on=False, color=meancol)


axin.plot(m[:, 0], m[:, 1], '--', lw=0.5, color=meancol)

# axin.text(0.7, 0.85, r'$\rho={:.3f}$'.format(meta_r[0, 1]), transform=axin.transAxes)
axin.text(0.65, -0.15, r'$\mu_p={:.3f}$'.format(means[1, 0]),
          transform=axin.transAxes, color=meancol)
axin.text(-0.05, 0.7, r'$\mu_r={:.3f}$'.format(means[1, 1]),
          transform=axin.transAxes, horizontalalignment='right', color=meancol)
# axin.text(0.7, 0.5, r'$\sigma_r={:.2f}$'.format(stdevs[1, 1]), transform=axin.transAxes)

tx = np.arange(axin_xlim[0]+0.002, axin_xlim[1]-0.001, 1e-4)
px = norm.pdf(tx, loc=means[1,0], scale=stdevs[1][0])
# axin.plot(tx, axin_ylim[0] + np.diff(axin_ylim)*0.03 + 0.7*1e-4* px, '-', clip_on=False, color='orange')
axin.plot(tx, axin_ylim[0] + 1e-4* px, '-', clip_on=False, color='orange')

ty = np.arange(axin_ylim[0]+0.033, axin_ylim[1]-0.01, 1e-3)
py = norm.pdf(ty, loc=means[1,1], scale=stdevs[1][1])
# axin.plot(axin_xlim[0] + np.diff(axin_xlim)*0.03 + 0.7*1e-4* py, ty, '-', clip_on=False, color='orange')
axin.plot(axin_xlim[0] + 0.7*1e-4* py, ty, '-', clip_on=False, color='orange')


# ft, axt = plt.subplots(1,1)
# axt.plot(ty, py)
# axt.plot(means[1,1], 0, 'o')
# plt.show()
# exit()

ax.legend()
ax.set_xlabel(r"$r_\mathrm{prediction}^2$")
ax.set_ylabel(r"$r_\mathrm{reconstruction}^2$")

# print(p)


ax.set_ylim(-.1, 1)
ax.set_xlim(0.8, 1)
axin.set_xlim(axin_xlim)
axin.set_ylim(axin_ylim)
ax.indicate_inset_zoom(axin, edgecolor="black")
# axin.set_xticks([])
# axin.set_yticks([])

plt.savefig("./resfigure/mapper-maco.eps")

# plt.show()
print('[OK]')