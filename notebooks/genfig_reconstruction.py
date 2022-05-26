import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.preprocessing import scale, minmax_scale


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


cp = scale(cc_pred)
c = scale(cc_val)

T = 59


fig2, axs2 = plt.subplots(1, 2)

axs2[0].plot(minmax_scale(c[:T]), label="original", color='k', alpha=1, lw=2)
axs2[0].plot(minmax_scale(np.sign(rec_perform[0, 1]) * cp[:T]), label="reconstructed",
             linestyle='-', color='r', alpha=1, lw=1)

axs2[1].plot(minmax_scale(cc_val[:]), minmax_scale(cc_pred[:]), 'r.', alpha=0.2)
axs2[1].plot([0, 1], [0, 1], 'k--')
axs2[1].text(0.35, .9, r'$r^2={:.2f}$'.format(rec_perform[0, 1]**2), horizontalalignment='right')

axs2[0].set_xlabel(r'$t$ (simulation step)')
axs2[0].set_ylabel(r'$z$')

axs2[0].legend(loc="lower left")


axs2[1].set_xlabel(r'$z(t-1)$')
axs2[1].set_ylabel(r'$\hat{z}(t-1)$')
axs2[1].set_xlim([0, 1])
axs2[1].set_ylim([0, 1])

axs2[0].text(-0.22, 1.02, "A",
             fontsize=16,
             transform=axs2[0].transAxes)

axs2[1].text(-0.22, 1.02, "B",
             fontsize=16,
             transform=axs2[1].transAxes)


fig2.tight_layout(pad=1, h_pad=0, w_pad=1)

fig2.savefig("./resdata/reconstruction.pdf")
# plt.show()