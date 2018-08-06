import numpy as np
from metrics import accuracy_score

class LogisticRegression:
    """构造函数"""
    def __init__(self):
        """初始化Linear Regression模型"""
        self.coef_ = None
        self.interception_ = None
        self._theta = None

    def __sigmoid(self, t):
        return 1 / (1 + np.exp(-t))

    """梯度下降法求解"""
    def fit(self, X_train, y_train, eta = 0.01, n_iters = 1e4 ):
        """根据训练数据集X_train, y_train 训练Linear Regression模型"""
        assert X_train.shape[0] == y_train.shape[0],\
            "the size of X_train must be equal to y_train"
        X_b = np.hstack([np.ones((len(X_train), 1)),X_train])


        """代价函数"""
        def J(theta, X_b, y):
            y_hat = self.__sigmoid(X_b.dot(theta))
            try:
                return - np.sum(y*np.log(y_hat) + (1-y)*np.log((1-y_hat))) / len(y)
            except:
                return float('inf')

        """代价函数求导"""
        def dJ(theta, X_b, y):
            # res = np.empty(len(theta))
            # res[0] = np.sum(X_b.dot(theta) - y)
            # for i in range(1, len(theta)):
            #     res[i] = (X_b.dot(theta) - y).dot(X_b[:, i])
            # return res * 2 / len(X_b)
            #return X_b.T.dot(self._sigmoid(X_b.dot(theta)) - y) / len(y)
            return X_b.T.dot(self.__sigmoid(X_b.dot(theta)) - y) / len(X_b)

        """梯度下降"""
        def gradient_descent(X_b, y, initial_theta, eta, n_iters, epsilon=1e-8):
            theta = initial_theta
            i_iter = 0

            while i_iter < n_iters:
                gradient = dJ(theta, X_b, y)
                last_theta = theta
                theta = theta - eta * gradient

                if (abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon):
                    break
                i_iter += 1
            return theta

        initial_theta = np.zeros(X_b.shape[1])
        self._theta = gradient_descent(X_b, y_train, initial_theta, eta, n_iters)
        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def predict_proba(self, X_predict):
        """给定待预测数据集X_predict ,给出相应结果集"""
        assert self.interception_ is not None and self.coef_ is not None, \
            "must fit before predict!"
        assert X_predict.shape[1] == len(self.coef_), \
            "the feture number of X_predict must be equal to X_train"

        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return self.__sigmoid(X_b.dot(self._theta))

    """预测函数"""
    def predict(self, X_predict):
        """给定待预测数据集X_predict ,给出相应结果集"""
        assert  self.interception_ is not None and self.coef_ is not None,\
            "must fit before predict!"
        assert X_predict.shape[1] == len(self.coef_), \
            "the feture number of X_predict must be equal to X_train"

        proba = self.predict_proba(X_predict)

        return np.array(proba>=0.5, dtype='int')

    """评价函数"""
    def score(self, x_test, y_test):
        y_predict = self.predict(x_test)
        return accuracy_score(y_test, y_predict)


    """返回一个可以用来表示对象的可打印字符串"""
    def __repr__(self):
        return "LogisticRegression()"