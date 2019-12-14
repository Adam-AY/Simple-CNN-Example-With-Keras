import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.datasets import mnist
import keras

import numpy as np
import matplotlib.pyplot as plt 

# Get the data
(x_train, y_train), (x_test,y_test) = mnist.load_data()

#normalize the data [0, 1]
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

#build the model
model = Sequential()
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

#training
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train,epochs=2)


#testing our model
model.save('epic_nim_reader.model')
new_model = tf.keras.models.load_model('epic_nim_reader.model')
predictions = new_model.predict(x_test)
print(predictions)

print(np.argmax(predictions[7]))

plt.imshow(x_test[7], cmap = plt.cm.binary)
plt.show()