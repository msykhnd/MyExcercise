import datetime
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D

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
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        self.function = Function()

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = self.function.sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
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
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}

        batch_num = x.shape[0]

        # forward
        a1 = np.dot(x, W1) + b1
        z1 = self.function.sigmoid(a1)
        y = np.dot(z1, W2) + b2
        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)
        dz1 = np.dot(dy, W2.T)

        da1 = self.function.sigmoid_grad(a1) * dz1
        grads['W1'] = np.dot(x.T, da1)
        grads['b1'] = np.sum(da1, axis=0)
        return grads

if __name__ == '__main__':
    # target = Nonliner_function()
    # X,Y = target.samplimg(2,1)
    # Z = target.function2(X, Y)
    # print(np.pi)


    # Generating Data Set
    target = Nonliner_function()
    X, Y = target.samplimg(100, 10)
    Z = target.function1(X, Y)
    All_data = target.get_coordinates(X, Y, Z)
    x0_input = np.ones(All_data.shape[0]).reshape((All_data.shape[0],1))
    print(All_data)
    All_data = np.hstack((x0_input,All_data))
    print(All_data.shape)

    fig = plt.figure()
    ax = Axes3D(fig)
    target.get_coordinates(X,Y,Z)
    ax.plot_wireframe(X, Y, Z)


    # Separating to train & Test (Random sampling)
    np.random.shuffle(All_data)
    Dataset_list = np.split(All_data, 4)
    Test_Data = Dataset_list[0]
    Train_Data = np.vstack(Dataset_list[1:])
    Test_input, Test_label = np.hsplit(Test_Data, [3])[0], np.hsplit(Test_Data, [3])[1]
    Train_input, Train_label = np.hsplit(Train_Data, [3])[0], np.hsplit(Train_Data, [3])[1]

    # Main Parameters
    iters_num = 10000
    train_size = Train_Data.shape[0]
    batch_size = 1
    learning_rate = 0.1
    train_loss_list = []

    network = ThreeLayerNet(input_size=3, hidden_size=10, output_size=1)

    iter_per_epoch = max(train_size / batch_size, 1)

    for i in range(iters_num):
        batch_mask = np.random.choice(train_size, batch_size)
        input_batch = Train_input[batch_mask]
        label_batch = Train_label[batch_mask]
        # print(input_batch)
        # print(label_batch)

        grad = network.gradient(input_batch, label_batch)

        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= learning_rate * grad[key]

        if (i % 100) == 0:
            loss = network.loss(Test_input, Test_label)
            print(loss)

    end = datetime.datetime.now()
    start = datetime.datetime.now()
    plt.show()