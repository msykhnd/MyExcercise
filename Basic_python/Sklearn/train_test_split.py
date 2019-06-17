import numpy as np
from sklearn.model_selection import train_test_split

X = np.arange(10).reshape((5, 2))
Y = np.array([1,1,0,1,0])
print(X.shape,Y.shape)

for i in range(Y.shape[0]):
    print(X[i],Y[i])

print("===")

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4)
for i in range(Y_train.shape[0]):
    print(X_train[i],Y_train[i])
print("===")

for i in range(Y_test.shape[0]):
    print(X_test[i],Y_test[i])
print("===")

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, shuffle=False)
for i in range(Y_train.shape[0]):
    print(X_train[i],Y_train[i])
print("===")

for i in range(Y_test.shape[0]):
    print(X_test[i],Y_test[i])
print("===")
