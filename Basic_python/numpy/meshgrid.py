import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

X, Y = np.meshgrid(np.arange(0, 10, 1), np.arange(0, 100, 10))

print(X, Y)
print(X.ravel())

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_wireframe(X, Y, np.sin(X + Y))
plt.show()
