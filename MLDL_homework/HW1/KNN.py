import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_result(train_data, test_data, labels, K):
    disMat = (np.sum((train_data - test_data) ** 2, axis=1))
    idx = labels[disMat.argsort()][:K]
    return pd.Series(idx).mode()[0]


def KNN(attrSize, K, data):
    # shuffle
    np.random.shuffle(data)

    # 70% 训练数据，30% 测试数据
    rate = .7
    train_data = data[:int(len(data) * rate)]
    test_data = data[int(len(data) * rate):]

    # train start
    error = 0
    for i in test_data:
        from_knn = get_result(train_data[:, :attrSize], i[:attrSize], train_data[:, attrSize], K)
        curr = i[attrSize]
        if from_knn != curr:
            error += 1
    acc_rate = 1.0 - error * 1.0 / len(test_data)
    return acc_rate


if __name__ == '__main__':
    iris_path = 'iris.txt'
    data = pd.read_csv(iris_path, header=None).to_numpy()

    attrSize = 4
    K_high = 30
    count = 30

    # 归一化处理
    minV = np.min(data[:, :attrSize], axis=0)
    maxV = np.max(data[:, :attrSize], axis=0)
    data[:, :attrSize] = (data[:, :attrSize] - minV) / maxV

    graph_list = []
    for i in range(1, K_high):
        acc_rate = 0.0
        for j in range(count):
            acc_rate += KNN(attrSize, i, data)
        acc_rate /= count * 1.0
        graph_list.append([i, acc_rate])
        print('{} acc rate: {}'.format(i, acc_rate))
    graph_list = np.array(graph_list)
    plt.plot(graph_list[:, 0], graph_list[:, 1])
    plt.show()
