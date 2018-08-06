import numpy as np
from math import sqrt

'''预测的准确度'''
def accuracy_score(y_true, y_predict):
    """计算y_true和y_predict之间的准确率"""
    assert y_true.shape[0] == y_predict.shape[0], \
        "the size of y_true must be equal to the size of y_predict"

    return  sum(y_true == y_predict) / len(y_true)

'''均方误差'''
def mean_squared_error(y_true, y_predict):
    """计算y_true和y_predict之间的ＭＳＥ"""
    assert len(y_true) == len(y_predict), \
        "the size of y_true must be equal to the size of y_predict"

    return np.sum((y_true - y_predict) ** 2) / len(y_true)

'''均方根误差'''
def root_mean_squared_error(y_true, y_predict):
    """计算y_true和y_predict之间的RMSE"""
    return sqrt(mean_squared_error(y_true, y_predict))

'''平均绝对误差'''
def mean_absolute_error(y_true, y_predict):
    """计算y_true和y_predict之间的MAE"""
    assert len(y_true) == len(y_predict), \
        "the size of y_true must be equal to the size of y_predict"
    return np.sum(np.absolute(y_true - y_predict)) / len(y_true)

'''R2：消除了对于不同样本，训练集大小不同的问题'''
def r2_score(y_true, y_predict):
    """计算y_true和y_predict之间的 R Square"""
    return 1 - mean_squared_error(y_true, y_predict) / np.var(y_true)

def TN(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 0) & (y_predict == 0))

def FP(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 0) & (y_predict == 1))

def FN(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 1) & (y_predict == 0))

def TP(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 1) & (y_predict == 1))

def confusion_matrix(y_true,y_predict):
    return np.array([
        [TN(y_test,y_predict),FP(y_test,y_predict)],
        [FN(y_test,y_predict),TP(y_test,y_predict)]
    ])

def precision_score(y_true,y_predict):
    tp = TP(y_true, y_predict)
    fp = FP(y_true, y_predict)
    try:
        return tp / (tp + fp)
    except:
        return 0.0
def recall_score(y_true,y_predict):
    tp = TP(y_true, y_predict)
    fn = FN(y_true, y_predict)
    try:
        return tp / (tp + fn)
    except:
        return 0.0

def f1_score(precision, recall):
    try:
        return (2 * precision * recall) / (precision + recall)
    except:
        return 0.0
def TPR(y_true, y_predict):
    tp = TP(y_true,y_predict)
    fn = FN(y_true,y_predict)
    try:
        return tp / (tp + fn)
    except:
        return 0.
def FPR(y_true, y_predict):
    fp = FP(y_true, y_predict)
    tn = TN(y_true, y_predict)
    try:
        return fp / (fp + tn)
    except:
        return 0.
