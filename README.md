# Simple-CNN-Example-With-Keras

A demonstration of how CNN works by implementing a code that recognizes hand-written digits. We used MNIST dataset.

### Prerequisites

Install [python 3.7](https://www.python.org/downloads/release/python-370/)

### Installing
Installing needed libraries

```shell
pip install tensorflow
pip install keras
```
or
install [Anaconda distribution](https://www.anaconda.com/distribution/), it contains all needed libraries

## Dataset

The MNIST dataset is automatically downloaded by Keras

```python
# this will download the dataset
(x_train, y_train), (x_test,y_test) = mnist.load_data()
```
## How it works

- Firstly we normalize the data

```python 
#normalize the data [0, 1]
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
```

- Secondly, we build our model

```python
#build the model
model = Sequential()
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
```

- Then, we train the model we have just created

```python
#training
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train,epochs=2)
```

-Finally, testing our model

```python
#testing our model
model.save('epic_nim_reader.model')
new_model = tf.keras.models.load_model('epic_nim_reader.model')
predictions = new_model.predict(x_test)
print(predictions)
```

### Running

```shell
python main.py
```


## Built With

* [Tensorflow](https://www.tensorflow.org)
* [Keras](https://keras.io)


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc

