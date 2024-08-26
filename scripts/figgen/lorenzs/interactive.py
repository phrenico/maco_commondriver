import matplotlib.pyplot as plt
import numpy as np
import time


x = np.linspace(-1, 1, 100)
y = np.sin(x)

plt.ion()
plt.figure()
plt.show()

plt.xlim(-1, 1)
plt.ylim(-1, 1)

for i in range(len(x)):
    plt.plot(x[i], y[i], 'b.')
    plt.draw()

    plt.pause(0.001)





time.sleep(1)