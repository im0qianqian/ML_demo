from sklearn.datasets import load_iris
from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    iris = load_iris()

    rate = .7  # 训练数据比例
    max_depth = 20  # 设置决策树最大深度 [1,20 - 1]

    # shuffle
    idx = np.random.permutation(len(iris.data))
    data = iris.data[idx]
    target = iris.target[idx]

    # 归一化处理
    minV = np.min(data, axis=0)
    maxV = np.max(data, axis=0)
    data = (data - minV) / maxV

    # 分割数据
    train_x = data[:int(len(iris.data) * rate)]
    train_y = target[:int(len(iris.data) * rate)]
    test_x = data[int(len(iris.data) * rate):]
    test_y = target[int(len(iris.data) * rate):]

    graph_list = []
    for i in range(1, max_depth):
        clf = tree.DecisionTreeClassifier(max_depth=i)
        clf.fit(train_x, train_y)
        # 预测并计算正确的结果个数
        s = sum(clf.predict(test_x) == test_y)
        acc_rate = s * 1.0 / len(test_y)
        graph_list.append([i, acc_rate])
        print('acc rate: ', acc_rate, ' max depth: ', i)
    graph_list = np.array(graph_list)
    plt.plot(graph_list[:, 0], graph_list[:, 1])
    plt.show()
