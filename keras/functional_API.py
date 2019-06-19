from keras import Model
from keras import Input,layers

input_tensor = Input(shape=(64,))
x = layers.Dense(32,activation='relu')(input_tensor)
x = layers.Dense(32,activation='relu')(x)
output_tensor = layers.Dense(10,activation='softmax')(x)

model = Model(input_tensor,output_tensor)
model.summary()

model.compile(optimizer='rmsprop',loss='categorical_crossentropy')

import numpy as np
x_train = np.random.random((1000,64))
y_train = np.random.random((1000,10))
score = model.evaluate(x_train,y_train)



