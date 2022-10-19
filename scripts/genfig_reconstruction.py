"""Generates the figure with the hidden variable reconstruction performance

"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.preprocessing import scale, minmax_scale

plt.style.use('./figure_twocol_config.mplstyle')

# Load data
df = pd.read_csv('./resdata/mappercoach_res.csv')
x_pred = df['x_pred'].values
x_test = df['x_valid'].values
Y_test= df[["Y_1_valid", "Y_2_valid"]].values
cc_pred= df['cc_pred'].values
cc_val = df['cc_valid'].values
learnings = np.load('./resdata/learning_curves.npy')


# compute correlation
rec_perform = np.corrcoef(cc_val[:], cc_pred[:])
print(rec_perform)

cp = scale(cc_pred)
c = scale(cc_val)

T = 59


fig2, axs2 = plt.subplots(1, 2)

axs2[0].plot(minmax_scale(c[:T]), label="original", color='k', alpha=1, lw=2)
axs2[0].plot(minmax_scale(np.sign(rec_perform[0, 1]) * cp[:T]), label="reconstructed",
             linestyle='-', color='#9EE004', alpha=1, lw=1.5)

axs2[1].plot(scale(cc_val[:]), scale(np.sign(rec_perform[0, 1]) * cc_pred[:]), '.', alpha=0.2, color="#9EE004")
axs2[1].plot([-1.9, 1.5], [-1.9, 1.5], 'k--')
axs2[1].text(.05, .9, r'$r^2={:.2f}$'.format(rec_perform[0, 1]**2), transform=axs2[1].transAxes)


axs2[0].set_xlabel(r'$t$ (simulation step)')
axs2[0].set_ylabel(r'$z$')

axs2[0].legend(loc="lower left")


axs2[1].set_xlabel(r'normalized $z(t-1)$')
axs2[1].set_ylabel(r'normalized $\hat{z}(t-1)$')
axs2[1].set_xlim([-1.9, 1.5])
axs2[1].set_ylim([-1.9, 1.5])
# axs2[1].set_ylim(axs2[1].get_xlim())
# axs2[1].set_xlim(axs2[1].get_ylim())


axs2[0].text(-0.22, 1.02, "A",
             fontsize=10,
             transform=axs2[0].transAxes)

axs2[1].text(-0.22, 1.02, "B",
             fontsize=10,
             transform=axs2[1].transAxes)


# fig2.tight_layout(pad=1, h_pad=0, w_pad=1)

fig2.savefig("./resfigure/reconstruction.eps")
# fig2.savefig("./resfigure/reconstruction.png")
# plt.show()