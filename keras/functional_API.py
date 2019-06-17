from keras.models import Sequential, Model
from keras import Input, layers

import numpy as np

##Sequential model
# seq_model = Sequential()
# seq_model.add(layers.Dense(32, activation='relu', input_shape=(64,)))
# seq_model.add(layers.Dense(32, activation='relu'))
# seq_model.add(layers.Dense(10, activation='softmax'))

##Functional API model
input_tensor = Input(shape=(64,))
x = layers.Dense(32, activation='relu')(input_tensor)
x = layers.Dense(32, activation='relu')(x)
output_tensor = layers.Dense(10, activation='softmax')(x)
functional_model = Model(input_tensor, output_tensor)

# seq_model.summary()
functional_model.summary()
functional_model.compile(optimizer='Adam',loss='categorical_crossentropy')

x_train = np.random.random((1000, 64))
y_train = np.random.random((1000, 10))

functional_model.fit(x_train, y_train)

score = functional_model.evaluate(x_train,y_train)
print(score)