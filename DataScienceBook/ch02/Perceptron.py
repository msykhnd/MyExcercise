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
        '''X:shape=[サンプル数，特徴量]のndarray
         y:shape=[サンプル数]
         クラスを返す
        '''
        # マーセツイスター生成器
        rgen = np.random.RandomState(self.random_state)
        # 一次元配列，適合後重み  gen.normalで生成，初期化
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        # 各エポックでの誤分類（更新）の数
        self.errors_ = []
        for _ in range(self.n_iter):  # Epoch 毎
            errors = 0
            for xi, target in zip(X, y):  # X，yの要素で更新
                update = self.eta * (target - self.predict(xi))
                # 重みW_1 ～ W_m の更新
                self.w_[1:] += update * xi
                # 重みW_0 の更新
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)


## 決定境界可視化関数

def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ['s', 'x', 'o', '~', 'v']
    colors = ['red', 'blue', 'lightgreen', 'gray', 'cyan']
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # 二つの特徴量から最大値と最小値を求める
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # グリッド配列の生成
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    # パーセプトロンにグリッドポイントを投げる
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    plt.contour(xx1, xx2, Z, alpha=0.3, cmap=cmap)

    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolors='black')


if __name__ == '__main__':
    # v1 = np.array([1, 2, 3])
    # v2 = 0.5 * v1
    # np.arccos(v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    # np.arccos 逆余弦関数 linalg ベクトルの長さを算出

    df = pd.read_csv('https://archive.ics.uci.edu/ml/'
                     'machine-learning-databases/iris/iris.data', header=None)
    print(type(df.tail()))
    print(df.tail())# 最後の五行だけ出力

    ## select setosa and versicolor
    # pd.df.iloc 複数のデータを範囲指定（行・列番号）で抽出する
    # 1～100行，4要素目(目的変数)を抽出
    y = df.iloc[0:100, 4].values
    print(y)
    # "Iris-setosa"を-1 それ以外(verginica)を1に変換
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

    plt.savefig('images/02_06.png', dpi=300)
    # plt.show()

    ppn = Perceptoron(eta=0.1, n_iter=10)
    ppn.fit(X, y)

    plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of update')
    plt.show()

    plot_decision_regions(X, y, classifier=ppn)
    plt.xlabel('sepal length')
    plt.ylabel('petal length')
    plt.legend(loc='upper left')
    plt.show()
