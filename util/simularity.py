# coding=UTF-8
# 相似度计算的工具类

import numpy as np
import pandas as pd
from collections import Counter
import math


# 计算样本xi~xj的“闵可夫斯基距离”
def calc_mk_dist(xi, xj, p):
    # xi, xj 入参为numpy数组
    if xi.shape != xj.shape:
        print("样本xi，xj不是同一维度")
        return
    dist_mk = 0.0
    z = np.ones((xi.shape[0], 1))
    dist_mk = np.power(np.dot(np.power(xi - xj, p), z), 1 / p)
    return dist_mk[0]


# 计算样本xi~xj的“欧氏距离”
def calc_ed_dist(xi, xj):
    dist_ed = 0.0
    # 欧氏距离为：p=2时的闵可夫斯基距离
    dist_ed = calc_mk_dist(xi, xj, 2)
    return dist_ed


# 计算样本xi~xj的“曼哈顿距离”
def calc_man_dist(xi, xj):
    dist_man = 0.0
    # 曼哈顿距离为：p=1时的闵可夫斯基距离
    dist_man = calc_mk_dist(xi, xj, 1)
    return dist_man


# 计算样本xi~样本集合Dj的“闵可夫斯基距离”,返回是一个距离数组
def calc_mk_dist_mat(xi, Dj, p):
    # xi, xj 入参为numpy数组
    if xi.shape[0] != Dj.shape[1]:
        print("样本xi，xj不是同一维度")
        return
    dist_mk = 0.0
    z = np.ones((xi.shape[0], 1))
    dist_mk = np.power(np.dot(np.power(xi - Dj, p), z), 1 / p)
    return np.ravel(dist_mk)


# 计算样本xi~Dj的“曼哈顿距离”
def calc_man_dist_mat(xi, Dj):
    dist_man = 0.0
    # 曼哈顿距离为：p=1时的闵可夫斯基距离
    dist_man = calc_mk_dist_mat(xi, Dj, 1)
    return dist_man


# 计算样本xi~Dj的“欧氏距离”
def calc_ed_dist_mat(xi, Dj):
    dist_ed = 0.0
    # 欧氏距离为：p=2时的闵可夫斯基距离
    dist_ed = calc_mk_dist_mat(xi, Dj, 2)
    return dist_ed


# 计算样本集X的任意两个样本之间的距离矩阵
def calc_dist_mat(X):
    m, n = X.shape
    D = np.zeros((m, m))  # 距离矩阵初始化
    for i in range(m):
        # 减少一层循环，提高距离矩阵计算效率
        D[i, :] = calc_ed_dist_mat(X[i], X)  # 计算欧式距离
        # for j in range(i + 1, m):
        #     D[i, j] = self.calc_ed_dist(X[i], X[j])  # 计算欧式距离
        #     D[j, i] = D[i, j]
    return D


# 找到xi_testsets的k个近邻，返回矩阵
# 注意xi_testsets一定不在X中
# 距离计算采用欧氏距离
def getKNN(xi_testset, X, Y, k):
    if pd.isnull(k):
        k = 3
    dist_mat = calc_ed_dist_mat(xi_testset, X)
    idx = dist_mat.argsort()
    # 如果xi_testsets也在X中，则距离最小的一定是xi_testsets本身，此时计算距离为0，所以去掉排在最前面的一个
    if xi_testset in X:
        idx = idx[1:-1]
    dist_mat = dist_mat[idx]
    Y = Y[idx]
    return dist_mat[0:k], Y[0:k]


# 找到xi_testsets的猜中近邻xi_nh和猜错近邻xi_nm，返回这两个样本
# 注意xi_testsets一定不在X中
# 该方法只适用二分类
# 距离计算采用欧氏距离
def getNHM(xi_testset, yi, X, Y):
    dist_mat = calc_ed_dist_mat(xi_testset, X)
    idxs = dist_mat.argsort()
    # 如果xi_testsets也在X中，则距离最小的一定是xi_testsets本身，此时计算距离为0，所以去掉排在最前面的一个
    if xi_testset in X:
        idxs = idxs[1:-1]
    dist_mat = dist_mat[idxs]
    xi_nh = np.zeros((xi_testset.shape))
    xi_nm = np.zeros((xi_testset.shape))
    for idx in idxs:
        if yi == Y[idx] and np.sum(xi_nh) == 0:
            xi_nh = X[idx, :]
        elif yi != Y[idx] and np.sum(xi_nm) == 0:
            xi_nm = X[idx, :]
        if np.sum(xi_nh) > 0 and np.sum(xi_nm) > 0:
            break
    return xi_nh, xi_nm


# 找到xi_testsets的猜中近邻xi_nh和猜错近邻xi_nm，返回这两个样本
# 注意xi_testsets一定不在X中
# 该方法适用于多分类
# 距离计算采用欧氏距离
def getNHM_F(xi_testset, yi, X, Y):
    labels = dict(Counter(Y))
    dist_mat = calc_ed_dist_mat(xi_testset, X)
    idxs = dist_mat.argsort()
    # 如果xi_testsets也在X中，则距离最小的一定是xi_testsets本身，此时计算距离为0，所以去掉排在最前面的一个
    if xi_testset in X:
        idxs = idxs[1:-1]
    dist_mat = dist_mat[idxs]
    xi_nh = np.zeros((xi_testset.shape))
    xi_nm = {}
    i = 0
    for idx in idxs:
        if yi == Y[idx] and np.sum(xi_nh) == 0:
            xi_nh = X[idx, :]
        elif yi != Y[idx] and Y[idx] not in xi_nm:
            xi_nm[Y[idx]] = X[idx, :]
            i += 1
        if np.sum(xi_nh) > 0 and i >= len(labels):
            break
    return xi_nh, xi_nm

# 余弦相似度计算
def calsim(l1, l2):
    a, b, c = 0.0, 0.0, 0.0
    for t1, t2 in zip(l1, l2):
        x1 = t1[1]
        x2 = t2[1]
        a += x1 * x2
        b += x1 * x1
        c += x2 * x2
    sim = a / math.sqrt(b * c) if not (b * c) == 0.0 else 0.0
    return sim


# 余弦相似度的计算公式：cos = X*Y / |X|*|Y| = sum(Xi*Yi)/(sqrt(sum(Xi*Xi))*Sqrt(sum(Yi*Yi)))
# 余弦值的范围在[-1,1]之间，值越趋近于1，代表两个向量的方向越接近；越趋近于-1，他们的方向越相反；接近于0，表示两个向量近乎于正交。
def calc_cos_sim(v1, v2):
    a, b, c = 0.0, 0.0, 0.0
    for x1, x2 in zip(v1, v2):
        a += x1 * x2
        b += x1 * x1
        c += x2 * x2
    sim = a / math.sqrt(b * c) if not (b * c) == 0.0 else 0.0
    return sim
