# -*- coding: utf -8 -*-

import os.path
import random
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb
from matplotlib import pyplot as plt
from sklearn import datasets  # 引入数据集,sklearn包含众多数据集
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # 数据标准化
from sklearn.decomposition import PCA, TruncatedSVD  # 主成分分析
from imblearn.over_sampling import SMOTE  # 上采样
from sklearn.model_selection import train_test_split, RepeatedKFold  # 将数据分为测试集和训练集
from sklearn.feature_selection import SelectFromModel, chi2, SelectKBest

from sklearn.neighbors import KNeighborsClassifier  # KNN
from sklearn.linear_model import LogisticRegression, Lasso  # Logistic
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier  # adaboost, 随机森林
from sklearn.tree import DecisionTreeClassifier  # 决策树
from sklearn.svm import SVC, LinearSVC  # SVM
from xgboost import XGBClassifier  # xgboost
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, roc_curve, \
    confusion_matrix  # 评分指标


def kFold_score(X, y, n_splits, n_repeats, model):
    sumP = 0
    sumR = 0
    sumF1 = 0
    sumA = 0
    sumAUC = 0
    sumFNR = 0
    sumFPR = 0
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats)

    for train_index, test_index in rkf.split(X, y):
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]

        # 训练模型
        model = model.fit(X_train, y_train)
        # 测试集性能
        y_pre = model.predict(X_test)
        y_score = model.predict_proba(X_test)[:, 1]

        P = precision_score(y_test, y_pre)
        R = recall_score(y_test, y_pre)
        F1 = f1_score(y_test, y_pre)
        A = accuracy_score(y_test, y_pre)
        AUC = roc_auc_score(y_test, y_score)

        cm = confusion_matrix(y_test, y_pre)
        # 从混淆矩阵中提取TP和FN的值
        TP = cm[1, 1]  # 实际为正例且预测为正例的样本数
        FN = cm[1, 0]  # 实际为正例但预测为负例的样本数
        # 计算真阳性率
        FNR = FN / (TP + FN)

        # 从混淆矩阵中提取FP和TN的值
        FP = cm[0, 1]  # 实际为负例但预测为正例的样本数
        TN = cm[0, 0]  # 实际为负例且预测为负例的样本数
        # 计算假阳性率
        FPR = FP / (FP + TN)

        sumP += P
        sumR += R
        sumF1 += F1
        sumA += A
        sumAUC += AUC
        sumFNR += FNR
        sumFPR += FPR
    P = format(sumP / (n_splits * n_repeats), '.4f')
    R = format(sumR / (n_splits * n_repeats), '.4f')
    F1 = format(sumF1 / (n_splits * n_repeats), '.4f')
    A = format(sumA / (n_splits * n_repeats), '.4f')
    AUC = format(sumAUC / (n_splits * n_repeats), '.4f')
    FNR = format(sumFNR / (n_splits * n_repeats), '.4f')
    FPR = format(sumFPR / (n_splits * n_repeats), '.4f')

    print(f'P:{P},R:{R},F1:{F1},A:{A},AUC:{AUC},FNR:{FNR},FPR:{FPR}')
    return A


def time_split(model, X, y):
    len = X.shape[0]
    split = int(len / 5)
    X_train, X_test, y_train, y_test = X[split:], X[:split], y[split:], y[:split]
    # 训练模型
    model = model.fit(X_train, y_train)
    # 测试集性能
    y_hat = model.predict(X_test)
    y_real = y_test
    P = precision_score(y_real, y_hat)
    R = recall_score(y_real, y_hat)
    F1 = f1_score(y_real, y_hat)
    A = accuracy_score(y_real, y_hat)
    AUC = roc_auc_score(y_real, y_hat)
    P = format(P, '.4f')
    R = format(R, '.4f')
    F1 = format(F1, '.4f')
    A = format(A, '.4f')
    AUC = format(AUC, '.4f')
    print(f'P:{P},R:{R},F1:{F1},A:{A},AUC:{AUC}')
    return P, R, F1, A, AUC


def random_split(X, y, model, random_state):
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state=random_state)
    # 训练模型
    model = model.fit(X_train, y_train)
    # 测试集性能
    y_pre = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]

    P = precision_score(y_test, y_pre)
    R = recall_score(y_test, y_pre)
    F1 = f1_score(y_test, y_pre)
    A = accuracy_score(y_test, y_pre)
    AUC = roc_auc_score(y_test, y_score)

    cm = confusion_matrix(y_test, y_pre)
    # 从混淆矩阵中提取TP和FN的值
    TP = cm[1, 1]  # 实际为正例且预测为正例的样本数
    FN = cm[1, 0]  # 实际为正例但预测为负例的样本数
    # 计算真阳性率
    FNR = FN / (TP + FN)

    # 从混淆矩阵中提取FP和TN的值
    FP = cm[0, 1]  # 实际为负例但预测为正例的样本数
    TN = cm[0, 0]  # 实际为负例且预测为负例的样本数
    # 计算假阳性率
    FPR = FP / (FP + TN)

    P = format(P, '.4f')
    R = format(R, '.4f')
    F1 = format(F1, '.4f')
    A = format(A, '.4f')
    AUC = format(AUC, '.4f')
    FNR = format(FNR, '.4f')
    FPR = format(FPR, '.4f')
    print(f'P:{P},R:{R},F1:{F1},A:{A},AUC:{AUC},FNR:{FNR},FPR:{FPR}')
    return P, R, F1, A, AUC


def result(model, X, y, random_state):
    # kFold_score(X, y, n_splits=5, n_repeats=4, model=model)
    # return time_split(model, X, y)
    random_split(X, y, model, random_state)


def getXy0(spl, df):
    # print(f"*******************FS*******************")

    X = df.iloc[:, :spl]
    y = df.iloc[:, -1]
    columns = X.columns  # 先保存特征名称
    # 标准化  # 归一化
    scaler = StandardScaler()
    # scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Lasso
    X = pd.DataFrame(X, columns=columns)
    lasso = Lasso(alpha=alpha)
    lasso.fit(X, y)
    # lasso.coef_相关系数列表
    coef = pd.DataFrame(lasso.coef_, index=X.columns)
    # 返回相关系数是否为0的布尔数组
    mask = lasso.coef_ != 0.0
    # print(coef[mask])
    # 对特征进行选择
    X = X.values
    X = X[:, mask]
    y = y.values
    # print(X.shape)
    return X, y


def getXy1(spl, n_components, df):
    # print(f"*******************WT*******************")

    X1 = df.iloc[:, spl:-1]
    y = df.iloc[:, -1]

    # # 标准化 归一化
    # scaler = StandardScaler()
    # # scaler = MinMaxScaler()
    # X1 = scaler.fit_transform(X1)

    # # 主成分分析压缩维度
    # pca = PCA(n_components=n_components, svd_solver='full')
    # X1 = pca.fit_transform(X1)

    X = X1.values
    y = y.values
    # print(X.shape)
    return X, y


def getXy2(spl, n_components, df):
    # print(f"*******************FS+BW*******************")
    X1, y1 = getXy0(spl, df)
    X2, y2 = getXy1(spl, n_components, df)
    X = np.concatenate((X1, X2), axis=1)
    # print(X.shape)
    return X, y1


def getXy3(years):
    print(f"*******************WA+WT+BS+BW*******************")

    df1 = pd.read_csv(f'data/1比1_Word2Vec_average_{years}years.csv', encoding='utf-8', dtype='float32')
    X1 = df.iloc[:, -201:-1]

    df2 = pd.read_csv(f'data/1比1_Word2Vec_tf-idf_{years}years.csv', encoding='utf-8', dtype='float32')
    X2 = df2.iloc[:, -201:-1]

    df3 = pd.read_csv(f'data/1比1_bert_sentences_{years}years.csv', encoding='utf-8', dtype='float32')
    X3 = df3.iloc[:, -769:-1]

    df4 = pd.read_csv(f'data/1比1_bert_words_{years}years.csv', encoding='utf-8', dtype='float32')
    X4 = df4.iloc[:, -769:-1] / 223

    X = pd.concat([X1, X2], axis=1)
    y = df1.iloc[:, -1]

    # 标准化 归一化
    scaler = StandardScaler()
    # scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # 主成分分析压缩维度
    pca = PCA(n_components=0.85, svd_solver='full')
    X = pca.fit_transform(X)

    y = y.values
    print(X.shape)
    return X, y


if __name__ == '__main__':
    if os.path.exists('output.txt'):
        open('output.txt', 'w')
    sys.stdout = open('output.txt', 'a')  # 重定向输出文件

    alpha = 0.005  # lasso L1系数
    n_components = 0.85  # pca保留特征信息量
    spl = -769  # 从哪截断,  word2vec 200维;  bert 768维; 词频 80维。
    # 数据集
    for i in range(1):
        years = 2
        filename = 'word_frequency_'
        if "Word2Vec" in filename:
            spl = -201
        elif "bert" in filename:
            spl = -769
        else:
            spl = -74
        path = f'data/1比1_{filename}{years}years.csv'
        df = pd.read_csv(path, encoding='utf-8', dtype='float32')
        # print(f"*******************{years}*******************")
        ls = [1]
        for target in ls:
            if target == 0:  # 数值
                X, y = getXy0(spl, df)
            elif target == 1:  # 文本
                X, y = getXy1(spl, n_components, df)
            elif target == 2:  # 结合
                X, y = getXy2(spl, n_components, df)
            else:  # 四种文本结合
                X, y = getXy3(years)

            for j in range(20):
                random.seed(10 * j + 1)
                random_state = random.randint(0, 100000)

                # logistic
                logistic = LogisticRegression(penalty="l2", max_iter=1000, solver='liblinear')
                # print('logistic:\t\t', end='')
                result(logistic, X, y, random_state)

                # adaboost
                adaboost = AdaBoostClassifier()
                # print('adaboost:\t\t', end='')
                result(adaboost, X, y, random_state)

                # 决策树
                decisionTree = DecisionTreeClassifier()
                # print('decisionTree:\t', end='')
                result(decisionTree, X, y, random_state)

                # 随机森林
                randomForest = RandomForestClassifier(n_estimators=200)
                # print('randomForest:\t', end='')
                result(randomForest, X, y, random_state)

                # importances = randomForest.feature_importances_
                # indices = np.argsort(importances)[::-1]  # 下标排序
                # for f in range(df.iloc[:, -74:-1].shape[1]):  # x_train.shape[1]=13
                #     print("%2d) %-*s %f" % \
                #           (f + 1, 30, df.iloc[:, -74:-1].columns[indices[f]], importances[indices[f]]))

                # SVM
                SVM = SVC(probability=True)  # cache_size 允许使用的内存大小
                # print('     SVM:\t\t', end='')
                result(SVM, X, y, random_state)

                # xgboost
                xgboost = XGBClassifier(n_estimators=200)
                # print('xgboost:\t\t', end='')
                result(xgboost, X, y, random_state)

                # lightgbm
                gbm = lgb.LGBMClassifier()
                # print('lightgbm:\t\t', end='')
                result(gbm, X, y, random_state)

                print("***")
