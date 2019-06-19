# from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds
import tensorflow.contrib.eager as tfe

from keras.layers import Dense, Flatten, Conv2D
from keras import Model

print(tf.__version__)

dataset, info = tfds.load('mnist', data_dir="tensorflow_datasets", with_info=True, as_supervised=True)
mnist_train, mnist_test = dataset['train'], dataset['test']

def convert_types(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255

    return image, label

mnist_train = mnist_train.map(convert_types).shuffle(10000).batch(32)
mnist_test = mnist_test.map(convert_types).batch(32)


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation="relu")
        self.flatten = Flatten()
        self.d1 = Dense(128, activation="relu")
        self.d2 = Dense(10, activation="softmax")

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)


if __name__ == '__main__':
    model = MyModel()
    loss_object = tf.keras.metrics.SparseCategoricalAccuracy()
    optimizer = tf.keras.optimizers.Adam()

    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_loss")

    test_loss = tf.keras.metrics.Mean(name="test_loss")
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="test_loss")


    def train_step(image, label):
        with tf.GradientTape() as tape:
            predictions = model()
            loss = loss_object(label, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_varibles))

        train_loss(loss)
        train_accuracy(label, predictions)


    def test_step(image, label):
        predictions = model(image)
        t_loss = loss_object(label, predictions)

        test_loss(t_loss)
        test_accuracy(label, predictions)


    epochs = 5
    for epoch in range(epochs):
        for image, label in mnist_train:
            train_step(image, label)
        for test_image, test_label in mnist_test:
            test_step(test_image, test_label)

        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch + 1, train_loss.result(), train_accuracy.result() * 100, test_loss.result(),
                              test_accuracy.result() * 100))



