import jieba
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.externals import joblib

#jieba分词
def chinese_word_cut(mytext):
    return " ".join(jieba.cut(mytext))


def getStop(text):
    f = open(text, 'r', encoding='utf8')
    res = f.readlines()
    f.close()
    return res

def predict(train_x,train_y,test_x,test_y):
    # 训练朴素贝叶斯分类器
    clf = MultinomialNB().fit(train_x, train_y)
    # 预测结果
    predicted = clf.predict(test_x)
    # 输出结果
    print(metrics.classification_report(test_y, predicted))
    print('accuracy_score: {}'.format(metrics.accuracy_score(test_y, predicted)))


if __name__ == '__main__':
    #导入数据
    train_data=pd.read_csv('cnews.train.txt',sep='\t',names=['label','content'])
    test_data=pd.read_csv('cnews.test.txt',sep='\t',names=['label','content'])
    print('train_data.info：')
    train_data.info()

    #先取前1/5的数据测试
    # train_data=train_data[:10000]
    # test_data=test_data[:2000]

    #添加分词
    train_data['content'] = train_data['content'].apply(chinese_word_cut)
    test_data['content'] = test_data['content'].apply(chinese_word_cut)

    #TF-IDF计算
    #计算词频
    vectorizer=CountVectorizer(max_features=100000,stop_words=getStop('stop_word.txt'))
    Xvalue=vectorizer.fit_transform(train_data['content'].append(test_data['content']))
    # word=vectorizer.get_feature_names()
    # print('word-------------------------')
    # print(word)
    # print('Xvalue.toarray----------------------------')
    # print(Xvalue.toarray())

    #统计每个分词的TF-IDF
    transformer=TfidfTransformer()
    tfidf=transformer.fit_transform(Xvalue)
    # print('tfidf-------------------------------------')
    # print(tfidf.toarray())

    #取训练和测试数据
    train_x = tfidf[:len(train_data)]
    train_y = train_data['label']
    test_x = tfidf[len(train_data):]
    test_y = test_data['label']

    #朴素贝叶斯预测
    predict(train_x, train_y, test_x, test_y)
