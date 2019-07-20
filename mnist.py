import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist

x_train = None
y_train = None
x_test = None
y_test = None


def init():
    global x_train, y_train, x_test, y_test
    (x_train_tmp, y_train_tmp), (x_test_tmp, y_test_tmp) = mnist.load_data()
    x_train = x_train_tmp.reshape(-1, 784)
    x_test = x_test_tmp.reshape(-1, 784)
    train_size = x_train.shape[0]
    test_size = x_test.shape[0]
    y_train = np.zeros((train_size, 10))
    for i in range(train_size):
        y_train[i][y_train_tmp[i]] = 1
    y_test = np.zeros((test_size, 10))
    for i in range(test_size):
        y_test[i][y_test_tmp[i]] = 1
    pass


if __name__ == '__main__':
    import time

    start_time = time.time()
    init()
    model = Sequential()
    model.add(Dense(units=1000, activation='sigmoid', input_dim=784))
    model.add(Dense(units=500, activation='sigmoid'))
    model.add(Dense(units=10, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=10, batch_size=1000)
    loss_and_metrics = model.evaluate(x_test, y_test, batch_size=1000)
    print(loss_and_metrics)
    print((time.time() - start_time))
