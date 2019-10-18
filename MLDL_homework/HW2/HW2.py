import pandas as pd
import jieba
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.externals import joblib


def chinese_word_cut(s):
    # 中文分词（jieba）
    return ' '.join(jieba.cut(s))


def train_and_predict(train_x, train_y, test_x, test_y):
    # 使用多项分布朴素贝叶斯分类器进行训练
    clf = MultinomialNB().fit(train_x, train_y)
    # 在本地保存 model 参数
    joblib.dump(clf, 'model.pkl')
    # 预测 test_x 中的结果
    predicted = clf.predict(test_x)
    # 打印报表
    print(metrics.classification_report(test_y, predicted))
    print('accuracy_score: {}'.format(metrics.accuracy_score(test_y, predicted)))


def read_vocab(path):
    # 读取 path 下文本的内容，返回 list https://github.com/goto456/stopwords
    f = open(path, 'r', encoding='utf8')
    res = f.readlines()
    f.close()
    return res


if __name__ == '__main__':
    start_time = time.time()
    # 加载训练集以及测试集
    print('start read csv...')
    train_data = pd.read_csv('cnews.train.txt', sep='\t', names=['label', 'content'])
    test_data = pd.read_csv('cnews.test.txt', sep='\t', names=['label', 'content'])

    # 对于数据集中每一个句子进行中文分词
    print('start chinese word cut...')
    train_data['content'] = train_data['content'].apply(chinese_word_cut)
    test_data['content'] = test_data['content'].apply(chinese_word_cut)

    print('start tfidf...')
    # 可以直接在 TfidfVectorizer 中传入停用词，设置词汇维度最大为 max_features
    tfidf = TfidfVectorizer(max_features=100000, stop_words=read_vocab('stop_word.txt'))
    # 拼接 train_data 以及 test_data 的作用是为了获得同一空间中的词汇向量（但实际中还是提前使用训练集做个映射表比较好，因为测试集中可能某些词汇在训练集中没有出现过）
    # 为了偷懒一起做方便很多
    x = tfidf.fit_transform(train_data['content'].append(test_data['content']))
    train_x = x[:len(train_data)]
    test_x = x[len(train_data):]
    train_y = train_data['label']
    test_y = test_data['label']

    # 输出 train_x[0] 中词语的 tf-idf 前 10 大的
    word = tfidf.get_feature_names()
    arg_sort = np.argsort(-train_x.toarray()[0])[:10]
    for i in arg_sort:
        print(word[i], train_x.toarray()[0][i])

    print('start train and predict...')
    train_and_predict(train_x, train_y, test_x, test_y)
    print('time: ', time.time() - start_time)
    print('The End.')
