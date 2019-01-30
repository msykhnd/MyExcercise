import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class Perceptoron(object):
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
            # X:shape=[サンプル数，特徴量]のndarray
            # y:shape=[サンプル数]
            # クラスを返す
        rgen = np.random.RandomState(self.random_state)
            #マーセツイスター生成器
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
            #一次元配列，適合後重み  gen.normalで生成，初期化
        self.errors_ = []
            #各エポックでの誤分類（更新）の数
        for _ in range(self.n_iter): # Epoch 毎
            errors = 0
            for xi, target in zip(X, y): # X，yの要素で更新
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)


if __name__ == '__main__':
    v1 = np.array([1, 2, 3])
    v2 = 0.5 * v1
    np.arccos(v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    df = pd.read_csv('https://archive.ics.uci.edu/ml/'
                     'machine-learning-databases/iris/iris.data', header=None)
    df.tail()

    # select setosa and versicolor
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)

    # extract sepal length and petal length
    X = df.iloc[0:100, [0, 2]].values

    # plot data
    plt.scatter(X[:50, 0], X[:50, 1],
                color='red', marker='o', label='setosa')
    plt.scatter(X[50:100, 0], X[50:100, 1],
                color='blue', marker='x', label='versicolor')

    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc='upper left')

    # plt.savefig('images/02_06.png', dpi=300)
    plt.show()

    pnn = Perceptoron