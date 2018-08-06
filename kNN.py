import numpy as np
from math import sqrt
from collections import Counter
from metrics import accuracy_score

class KNNClassifier:
    def __init__(self, k):
        """初始化kNN分类器"""
        assert k >= 1, "k must be valid"
        self.k = k
        self._X_train = None
        self._Y_train = None

    def fit(self, X_train, Y_train):
        """根据训练集数据，训练kNN分类器"""
        self._X_train = X_train
        self._Y_train = Y_train
        return self

    def predict(self, X_predict):
        """给定预测数据集X_predict, 返回X_predict的结果向量"""
        assert self._Y_train is not None and self._X_train is not None, \
            "must fit before predict!"
        assert X_predict.shape[1] == self._X_train.shape[1],\
            "the feature number of X_predict must be equals to X_train"

        y_predict = [self._predict(x) for x in X_predict]

        return np.array(y_predict)

    def _predict(self, x):
        """给定单个待预测数据x，返回x的预测结果"""
        assert x.shape[0] == self._X_train.shape[1],\
            "the feature number of x must be equals to X_train"

        distances = [sqrt(np.sum((x_train - x)**2)) for x_train in self._X_train]

        nearest = np.argsort(distances)

        topK_y = [self._Y_train[i] for i in nearest[:self.k]]

        votes = Counter(topK_y)
        return votes.most_common()[0][0]


    def score(self, X_test, y_test):
        """根据测试数据集X_test和y_test确定当前模型的准确度"""
        y_predict = self.predict(X_test)
        return accuracy_score(y_test, y_predict)

    def __repr__(self):
        return "KNN(k=%d)" % self.k