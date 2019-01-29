import gzip
import datetime
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = 784

mnist_files = [
    'train-images-idx3-ubyte.gz',
    'train-labels-idx1-ubyte.gz',
    't10k-images-idx3-ubyte.gz',
    't10k-labels-idx1-ubyte.gz'
]

class Mnist(object):
    def __init__(self, file_list=mnist_files):
        self.name = 'MNIST_dataset'
        self.data = dict()
        self.data['train_img'] = self.load_img(file_list[0])
        self.data['train_label'] = self.load_label(file_list[1])
        self.data['test_img'] = self.load_img(file_list[2])
        self.data['test_label'] = self.load_label(file_list[3])

    def load_img(self, file_name):
        print("Converting " + file_name + " to NumPy Array ...")
        with gzip.open(file_name, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, img_size)
        print("Done")
        return data

    def load_label(self, file_name):
        print("Converting " + file_name + " to NumPy Array ...")
        with gzip.open(file_name, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
        print("Done")
        return labels

    def img_show(self, data_name, num):
        img = self.data[data_name][num]
        pil_img = Image.fromarray(np.uint8(img.reshape(28, 28)))
        pil_img.show()


class ThreeLayerNet(object):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, weight_init_std=0.01):
        self.params = dict()
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size1)
        self.params['b1'] = np.zeros(hidden_size1)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size1, hidden_size2)
        self.params['b2'] = np.zeros(hidden_size2)
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size2, output_size)
        self.params['b3'] = np.zeros(output_size)

    def predict(self, input_data):
        w1, w2, w3 = self.params['W1'], self.params['W2'], self.params['W3']
        b1, b2, b3 = self.params['b1'], self.params['b2'], self.params['b3']
        f = Function()

        a1 = np.dot(input_data, w1) + b1
        z1 = f.sigmoid(a1)
        a2 = np.dot(z1, w2) + b2
        z2 = f.sigmoid(a2)
        a3 = np.dot(z2, w3) + b3
        y = f.softmax(a3)
        return y

    def lose(self, input_data, train_data):
        y = self.predict(input_data)
        if y.ndim == 1:
            train_data = train_data.reshape(1, train_data.size)
            y = y.reshape(1, y.size)

        if train_data.size == y.size:
            train_data = train_data.argmax(axis=1)
        batch_size = y.shape[0]
        return -np.sum(np.log(y[np.arange(batch_size), train_data])) / batch_size

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def gradient(self, input_data, t):
        w1, w2, w3 = self.params['W1'], self.params['W2'], self.params['W3']
        b1, b2, b3 = self.params['b1'], self.params['b2'], self.params['b3']
        p1 = 0.9
        p2 = 0.9
        p3 = 0.9
        dr1 = np.where(np.random.rand(input_data.shape[1]) < p1, True, False)
        dr2 = np.where(np.random.rand(b1.shape[0]) < p2, 1, 0)
        dr3 = np.where(np.random.rand(b2.shape[0]) < p3, 1, 0)

        grads = dict()
        batch_num = input_data.shape[0]
        f = Function()

        # forward
        z0 = input_data
        z0[:, np.where(dr1)] = 0
        a1 = np.dot(z0, w1) + b1
        z1 = f.sigmoid(a1) * dr2
        a2 = np.dot(z1, w2) + b2
        z2 = f.sigmoid(a2) * dr3
        a3 = np.dot(z2, w3) + b3
        y = f.softmax(a3)

        # backward
        dy = (y - t) / batch_num

        grads['W3'] = np.dot(z2.T, dy)
        grads['b3'] = np.sum(dy, axis=0)

        da2 = np.dot(dy, w3.T)
        dz2 = f.sigmoid_grad(a2) * da2

        grads['W2'] = np.dot(z1.T, dz2)
        grads['b2'] = np.sum(dz2, axis=0)

        da1 = np.dot(dz2, w2.T)
        dz1 = f.sigmoid_grad(a1) * da1

        grads['W1'] = np.dot(z0.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads


class Function(object):
    def __init__(self):
        self.name = ""

    def sigmoid(self, x):
        sigmoid_range = 34.538776394910684
        x[np.where(x > sigmoid_range)] = sigmoid_range
        x[np.where(x < -sigmoid_range)] = -sigmoid_range
        return 1 / (1 + np.exp(-x))

    def sigmoid_grad(self, x):
        return (1.0 - self.sigmoid(x)) * self.sigmoid(x)

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


def show_accuracy(train_acc_list, test_acc_list):
    markers = {'train': 'o', 'test': 's'}
    x = np.arange(len(train_acc_list))
    plt.plot(x, train_acc_list, label='train acc')
    plt.plot(x, test_acc_list, label='test acc', linestyle='--')
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
    plt.show()

if __name__ == '__main__':
    start = datetime.datetime.now()
    mnist = Mnist()
    f = Function()

    # mnist.img_show('train_img', 1)
    #print(mnist.data['train_label'][0])

    (train_img, train_label) = (mnist.data['train_img'], f.chnge_one_hot_label(mnist.data['train_label']))
    (test_img, test_label) = (mnist.data['test_img'], f.chnge_one_hot_label(mnist.data['test_label']))

    print('making 3layers neural net ...')
    network = ThreeLayerNet(input_size=784, hidden_size1=50, hidden_size2=50, output_size=10)
    print('Done')

    iters_num = 100000
    train_size = train_img.shape[0]
    batch_size = 100
    learning_rate = 0.1

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    iter_per_epoch = max(train_size / batch_size, 1)

    for i in range(iters_num):
        batch_mask = np.random.choice(train_size, batch_size)

        input_batch = train_img[batch_mask]
        label_batch = train_label[batch_mask]

        grad = network.gradient(input_batch, label_batch)

        for key in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3'):
            network.params[key] -= learning_rate * grad[key]

        loss = network.lose(input_batch, label_batch)
        train_loss_list.append(loss)

        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(train_img,train_label)
            test_acc = network.accuracy(test_img, test_label)

            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)

            print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

    end = datetime.datetime.now()
    print("start : " + start.strftime("%Y/%m/%d %H:%M:%S"))
    print("end   : " + end.strftime("%Y/%m/%d %H:%M:%S"))
    print("delta : " + str((end - start).seconds) + " sec")
    show_accuracy(train_acc_list,test_acc_list)
