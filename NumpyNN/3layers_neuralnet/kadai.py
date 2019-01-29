import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from statistics import mean

class Nonliner_function(object):
    def __init__(self, ):
        self.name = "Nonliner "

    def samplimg(self, xy_limit=1, interval=0.1):
        nparray_x = np.arange(-1 * xy_limit, xy_limit, interval)
        nparray_y = np.arange(-1 * xy_limit, xy_limit, interval)
        return np.meshgrid(nparray_x, nparray_y)

    def function1(self, nparray_x, nparray_y):
        Z = (1 + np.sin(4 * np.pi * nparray_x)) * nparray_y / 2
        return Z

    def function2(self, nparray_x, nparray_y):
        Z = nparray_x * nparray_y
        return Z

    def get_coordinates(self, X, Y, Z):
        x = X.ravel()
        y = Y.ravel()
        z = Z.ravel()
        xyz = np.hstack((x.reshape(len(x), 1), y.reshape(len(y), 1), z.reshape(len(z), 1)))
        return xyz


class Function(object):
    def __init__(self):
        self.name = "Function class"

    def liner(self, x):
        return x

    def sigmoid(self, x):

        return 1 / (1 + np.exp(-x))

    def sigmoid_grad(self, x):
        return (1.0 - self.sigmoid(x)) * self.sigmoid(x)

    def relu(x):
        return np.maximum(0, x)

    def relu_grad(x):
        grad = np.zeros(x)
        grad[x >= 0] = 1
        return grad

    def softmax(self, x):
        if x.ndim == 2:
            x = x.T
            x = x - np.max(x, axis=0)
            y = np.exp(x) / np.sum(np.exp(x), axis=0)
            return y.T
        x = x - np.max(x)
        return np.exp(x) / np.sum(np.exp(x))

    def chnge_one_hot_label(self, x):
        T = np.zeros((x.size, 10))
        for idx, row in enumerate(T):
            row[x[idx]] = 1
        return T


class ThreeLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 重みの初期化
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.function = Function()

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']

        a1 = np.dot(x, W1)
        z1 = self.function.sigmoid(a1)
        a2 = np.dot(z1, W2)
        y = self.function.liner(a2)

        return y

    # x:入力データ, t:教師データ
    def loss(self, x, t):
        y = self.predict(x)
        batch_size = y.shape[0]
        return np.sum((y-t)**2)/(2*batch_size)


    # x:入力データ, t:教師データ
    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        grads = {}

        batch_num = x.shape[0]

        # forward
        a1 = np.dot(x, W1)
        z1 = self.function.sigmoid(a1)
        y = np.dot(z1, W2)

        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        dz1 = np.dot(dy, W2.T)

        da1 = self.function.sigmoid_grad(a1) * dz1
        grads['W1'] = np.dot(x.T, da1)
        return grads


if __name__ == '__main__':
    # データセットの生成
    target = Nonliner_function()
    X, Y = target.samplimg(1, 0.01)
    Z = target.function1(X, Y)
    All_data = target.get_coordinates(X, Y, Z)
    Bios_1 = np.ones(All_data.shape[0]).reshape((All_data.shape[0], 1))
    All_data = np.hstack((Bios_1, All_data)) #データにバイアス（１）を追加
    np.random.shuffle(All_data) #データの順不同化

    # ターゲット関数の３Dプロット
    fig = plt.figure()
    ax = Axes3D(fig)
    target.get_coordinates(X, Y, Z)
    ax.plot_wireframe(X, Y, Z)

    # 主な設定項目
    Max_Epoch = 100
    train_size = All_data.shape[0]*3/4
    batch_size = 1000
    learning_rate = 0.1
    train_loss_list = []
    iter_per_epoch = max(train_size / batch_size, 1)
    iters_num = int(iter_per_epoch) * Max_Epoch

    # 各種表示
    print("Number of Sampled point", All_data.shape[0])
    print("Train data num", train_size)
    print("batch size",batch_size)
    print("Iter per Epoch", iter_per_epoch)
    print("Iter num",iters_num)
    # 交差検証のためのループ
    # データセットのTrain，Test分割（３：１）
    Dataset_list = np.split(All_data, 4)

    last_loss_list = []

    for j in range(4):
        # データセットリストの最初をテスト用，残りを学習用にする
        Test_Data = Dataset_list[0]
        Train_Data = np.vstack(Dataset_list[1:])
        Test_input, Test_label = np.hsplit(Test_Data, [3])[0], np.hsplit(Test_Data, [3])[1]
        Train_input, Train_label = np.hsplit(Train_Data, [3])[0], np.hsplit(Train_Data, [3])[1]

        # 次の交差検証のためにリスト内部をシフト
        Dataset_list = Dataset_list[1:]+Dataset_list[:1]

        # ネットワークの生成
        network = ThreeLayerNet(input_size=3, hidden_size=100, output_size=1)

        # 学習のためのループ
        for i in range(iters_num):
            # バッチ学習のためのランダム抽出
            batch_mask = np.random.choice(int(train_size), batch_size)
            input_batch = Train_input[batch_mask]
            label_batch = Train_label[batch_mask]

            # 勾配更新量の計算
            grad = network.gradient(input_batch, label_batch)

            # 重みの計算と更新
            for key in ('W1', 'W2'):
                network.params[key] -= learning_rate * grad[key]

            # 1エポック毎にテスト用データでLossを計算
            if (i % iter_per_epoch ) == 0:
                loss = network.loss(Test_input, Test_label)
                print("epoch:",i/iter_per_epoch,"Loss: {:.10f}".format(loss))

        last_loss_list.append(loss)

    print("Cross Validation loss{:.10f}".format(mean(last_loss_list)))
    # 最後に図を描画
    plt.show()