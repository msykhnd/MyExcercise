import numpy as np

a = np.arange(12)
print(a)

r1 = np.reshape(a, (-1, 3))
print(r1)

b = np.array([0, 1, 2, 10, 20, 30, 100, 200, 300, 1000, 2000, 3000])
r2 = np.reshape(b, (2, 2, 3))
print(r2)
# print(r2.T)
print(r2.reshape(3,2,2))