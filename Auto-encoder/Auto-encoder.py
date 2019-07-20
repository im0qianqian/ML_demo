import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist
from PIL import Image
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

x_train = None
y_train = None
x_test = None
y_test = None


def show_xy(point, labels):
    color = ['b', 'g', 'r', 'k', 'c', 'm', 'y', 'orange', 'purple', 'olive']
    x = point[:, 0]
    y = point[:, 1]
    for i in range(10):
        plt.scatter(x[labels == i], y[labels == i], c=color[i])


def show_xyz(point, labels):
    # 随机挑选 2000 个点展示着图上（唔，数据太大了卡呀
    choice = np.random.choice(len(labels), 2000)
    point = point[choice]
    labels = labels[choice]

    color = ['b', 'g', 'r', 'k', 'c', 'm', 'y', 'orange', 'purple', 'olive']
    x = point[:, 0]
    y = point[:, 1]
    z = point[:, 2]
    ax = plt.axes(projection='3d')

    for i in range(10):
        ax.scatter3D(x[labels == i], y[labels == i], z[labels == i], c=color[i])


def show_diff_image():
    """
    auto encoder show
    """
    global x_test
    np.random.shuffle(x_test)
    x_res = model.predict(x_test)
    res = None
    for i in range(15):
        tmp = None
        for j in range(5):
            idx = i * 15 + j
            old_image = x_test[idx].reshape(28, 28)
            new_image = x_res[idx].reshape(28, 28)
            if tmp is None:
                tmp = np.hstack((old_image, new_image))
            else:
                tmp = np.hstack((tmp, np.hstack((old_image, new_image))))
        if res is None:
            res = tmp
        else:
            res = np.vstack((res, tmp))
    image = Image.fromarray(res).convert('RGB')
    image.save('diff_image.png')
    image.show()


def show_decode_image(x_start, y_start, x_step, y_step, size):
    """
    展示在二维 code 下绘制的图像效果与各维度关系
    """
    res = None
    for i in range(size):
        tmp = None
        for j in range(size):
            x = x_start + x_step * i
            y = y_start + y_step * j
            image = model_lst.predict(np.array([[x, y]])).reshape(28, 28)
            if tmp is None:
                tmp = image
            else:
                tmp = np.hstack((tmp, image))
        if res is None:
            res = tmp
        else:
            res = np.vstack((res, tmp))
    image = Image.fromarray(res).convert('RGB')
    image.save('decode_image.png')
    image.show()


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28 * 28)
    x_test = x_test.reshape(-1, 28 * 28)

    model = Sequential()
    model_pre = Sequential()
    model_lst = Sequential()

    code_size = 2
    dense_1 = Dense(units=1000, activation='relu', input_dim=784)
    model.add(dense_1)
    model_pre.add(dense_1)

    dense_2 = Dense(units=500, activation='relu')
    model.add(dense_2)
    model_pre.add(dense_2)

    dense_3 = Dense(units=code_size, activation='relu')
    model.add(dense_3)
    model_pre.add(dense_3)

    dense_4 = Dense(units=500, input_dim=code_size, activation='relu')
    model.add(dense_4)
    model_lst.add(dense_4)

    dense_5 = Dense(units=1000, activation='relu')
    model.add(dense_5)
    model_lst.add(dense_5)

    dense_6 = Dense(units=784, activation='relu')
    model.add(dense_6)
    model_lst.add(dense_6)

    model.compile(optimizer='adam', loss='mean_squared_error')
    model_pre.compile(optimizer='adam', loss='mean_squared_error')
    model_lst.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(x_train, x_train, epochs=50, batch_size=500)

    if code_size == 3:
        show_xyz(model_pre.predict(x_test), y_test)
    elif code_size == 2:
        show_xy(model_pre.predict(x_test), y_test)
        show_decode_image(1000, 1000, 100, 100, 30)
    show_diff_image()
