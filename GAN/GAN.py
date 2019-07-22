import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.datasets import mnist
from keras.optimizers import Adam
from PIL import Image
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt


def create_model():
    generator_model = Sequential()
    generator_model.add(Dense(units=1000, input_dim=10, activation='relu'))
    # generator_model.add(Dense(units=500, activation='relu'))
    generator_model.add(Dense(units=28 * 28, activation='tanh'))

    discriminator_model = Sequential()
    discriminator_model.add(Dense(units=1000, input_dim=28 * 28, activation='relu'))
    # discriminator_model.add(Dense(units=500, activation='relu'))
    discriminator_model.add(Dense(units=1, activation='sigmoid'))

    model = Sequential()
    for i in generator_model.layers:
        model.add(i)
    for i in discriminator_model.layers:
        model.add(i)

    generator_model.compile(optimizer=Adam(0.0001), loss='mean_squared_error')
    discriminator_model.compile(optimizer=Adam(0.0001), loss='mean_squared_error')

    return model, generator_model, discriminator_model


def get_generation(model, size=10000):
    x = model.predict(np.random.random((size, 10)))
    x = (x + 1) / 2.0  # 因为使用了 tanh 激活函数，所以原始范围为 [-1,1]
    y = np.zeros((size, 1))
    return x, y


def train_discriminator(generator_model, discriminator_model):
    def get_data():
        x = mnist_test.reshape(-1, 28 * 28) / 255.0
        y = np.ones((len(x), 1))
        tmp_x, tmp_y = get_generation(generator_model, len(x))
        return np.vstack((x, tmp_x)), np.vstack((y, tmp_y))

    x, y = get_data()
    for i in discriminator_model.layers:
        i.trainable = True

    idx = list(range(len(x)))
    np.random.shuffle(idx)

    discriminator_model.compile(optimizer=Adam(0.0001), loss='mean_squared_error')
    discriminator_model.fit(x[idx], y[idx], epochs=2, batch_size=1000)


def train_generator(model, discriminator_model):
    for i in discriminator_model.layers:
        i.trainable = False
    x = np.random.random((20000, 10))
    y = np.ones((len(x), 1))

    model.compile(optimizer=Adam(), loss='mean_squared_error')
    model.fit(x, y, epochs=5, batch_size=1000)


def show_image():
    x = np.random.random((15, 10))
    res = generator_model.predict(x)

    _range = res.max() - res.min()
    res = (res - res.min()) / _range * 255
    tmp = None
    for i in res:
        if tmp is None:
            tmp = i.reshape(28, 28)
        else:
            tmp = np.vstack((tmp, i.reshape(28, 28)))
    image = Image.fromarray(tmp).convert('RGB')
    image.show()


def train_iteration(model, generator_model, discriminator_model):
    train_discriminator(generator_model, discriminator_model)
    train_generator(model, discriminator_model)


def start(model, generator_model, discriminator_model):
    for i in range(50):
        train_iteration(model, generator_model, discriminator_model)
        if i % 5 == 0:
            show_image()


if __name__ == '__main__':
    (mnist_train, _), (mnist_test, _) = mnist.load_data()
    model, generator_model, discriminator_model = create_model()
