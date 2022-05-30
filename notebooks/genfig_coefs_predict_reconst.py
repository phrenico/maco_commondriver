import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load Data
rdf = pd.read_csv('./resdata/r_values.csv', index_col=0)
rs = rdf.values

# Plot the coeficient of determinations of prediction and reconstruction against each other
plt.figure()

plt.plot(rs[:, 0]**2, rs[:, 1]**2, 'o')

plt.xlabel(r"$r_\mathrm{prediction}^2$")
plt.ylabel(r"$r_\mathrm{reconstruction}^2$")

plt.ylim(-.1, 1)
plt.xlim(0.8, 1)

plt.savefig("./resfigure/mapper-maco.pdf")

# plt.show()