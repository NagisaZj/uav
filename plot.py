import matplotlib.pyplot as plt
import numpy as np

res = np.load("result.npy")
plt.plot(res)
plt.ylabel('Moving average ep reward')
plt.xlabel('Step')
plt.show()