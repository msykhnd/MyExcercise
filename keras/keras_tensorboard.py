from keras import Model
from keras import Input, layers

def make_model(name):
    input_tensor = Input(shape=(64,))
    x = layers.Dense(32, activation='relu')(input_tensor)
    x = layers.Dense(32, activation='relu')(x)
    output_tensor = layers.Dense(10, activation='softmax')(x)

    model = Model(input_tensor, output_tensor)
    model.summary()

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    model.name=name
    return model

import numpy as np

x_train = np.random.random((1000, 64))
y_train = np.random.random((1000, 10))

x_test = np.random.random((100, 64))
y_test = np.random.random((100, 10))

from keras.callbacks import TensorBoard

tb = TensorBoard(log_dir='log_dir', histogram_freq=1, embeddings_data=1)

model = make_model("test")
model.fit(x_train, y_train,epochs=10,validation_data=[x_test,y_test], callbacks=[tb])

score = model.evaluate(x_train, y_train)
