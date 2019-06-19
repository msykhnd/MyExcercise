import numpy as np

b = np.array([0, 1, 2, 11, 12, 13, 20, 21, 22, 31, 41,42])
r2 = np.reshape(b, (4, -1))
c = np.arange(0, 4).reshape((4,-1))

print(r2, r2.shape)
print(c, c.shape)

print(np.hstack((r2,c)))