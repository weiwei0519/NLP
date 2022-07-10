# coding=utf-8
# 条件随机场模型

import os
import sys
import pickle
import dictionary as dic
from collections import Counter
from math import exp, log
import numpy as np
from scipy.optimize import fmin_l_bfgs_b

STARTING_STATE = '*'  # STARTING_STATE
STARTING_INDEX = 0
SCALING_THRESHOLD = 1e250

ITERATION_NUM = 0
SUB_ITERATION_NUM = 0
TOTAL_SUB_ITERATIONS = 0
GRADIENT = None

if sys.getdefaultencoding() != 'utf-8':
    sys.setdefaultencoding('utf-8')


# 前向-后向算法(P225 #11.3.1)
def forward_backword_calc(Xi_len, prob_M, state_num):
    alpha = np.zeros((Xi_len, state_num))
    beta = np.zeros((Xi_len, state_num))
    scaling_dic = dict()
    alpha[0, :] = prob_M[0][0, :]
    t = 1
    while t < Xi_len:
        scaling_i = None
        scaling_coefficient = None
        overflow_occured = False
        alpha[t, :] = np.dot(alpha[t - 1, :], prob_M[t])
        if len(np.where(alpha[t, :] > SCALING_THRESHOLD)[0]) > 0 or len(
                np.where(alpha[t, :] >= float('inf'))[0]) > 0:
            # 计算溢出double_scalars
            overflow_occured = True
            scaling_i = t - 1
            scaling_coefficient = SCALING_THRESHOLD
            scaling_dic[scaling_i] = scaling_coefficient
        if overflow_occured:
            alpha[t - 1, :] /= scaling_coefficient
            alpha[t, :] = 0
        else:
            t += 1

    t = Xi_len - 1
    beta[t, :] = 1.0
    t -= 1
    while t >= 0:
        beta[t, :] = np.dot(beta[t + 1, :], prob_M[t + 1])
        if t in scaling_dic.keys():
            beta[t, :] /= scaling_dic[t]
        t -= 1

    Z_Xi = sum(alpha[Xi_len - 1])
    return alpha, beta, Z_Xi, scaling_dic


# 基于全局状态转移矩阵，观察状态矩阵，构建每个训练样本Xi的M矩阵，P223
def build_probability_matrix(omega, feature_dic, Xi_feature, state_num):
    # prob_M[i][pre_y, y]
    prob_M = list()
    t = 0
    while t < len(Xi_feature):
        feature_t = Xi_feature[t]
        prob_M_t = np.zeros((state_num, state_num))
        for (pre_y, y, x_observe_name), _ in feature_t.items():
            feature_id = feature_dic[(pre_y, y, x_observe_name)]
            score = omega[feature_id]
            if pre_y == -1:
                prob_M_t[:, y] += score
            else:
                prob_M_t[pre_y, y] += score
        prob_M_t = np.exp(prob_M_t)
        if t == 0:
            prob_M_t[STARTING_INDEX + 1:] = 0
        else:
            # 在句子中间时，不会出现STARTING_INDEX的状态
            prob_M_t[:, STARTING_INDEX] = 0
            prob_M_t[STARTING_INDEX, :] = 0
        prob_M.append(prob_M_t)
        t += 1
    return prob_M


# 计算 p(y[i] | x[i]) = exp(sum_t(w_t * feature_t)) / z(x[i])
def _conditional_probability(omega, state_dic, observe_dic, feature_dic, state_trans_matrix, observe_trans_matrix, i):
    feature_dic_i = feature_dic[i]
    state_trans_num = state_trans_matrix.shape[0] * state_trans_matrix.shape[1]
    observe_trans_num = observe_trans_matrix.shape[0] * observe_trans_matrix.shape[1]
    omega_state_trans = omega[0:state_trans_num].reshape(state_trans_matrix.shape[0], state_trans_matrix.shape[1])
    omega_observe_trans = omega[state_trans_num:].reshape(observe_trans_matrix.shape[0], observe_trans_matrix.shape[1])
    sum_k = 0
    multipy_t = 1
    # Z三维矩阵，元素为[t, pre_y, y]
    Z_matrix = np.zeros((len(feature_dic_i), len(state_dic), len(state_dic)))
    Z_prob = np.zeros((len(state_dic), len(state_dic)))
    for t, feature_dic_t in feature_dic_i.items():
        # 先计算转移特征，格式为：{i: {t: {y_trans:{feature_desc: count}, X_trans:{feature_desc: count}}}}
        for trans_feature_desc, _ in feature_dic_t['y_trans'].items():
            pre_y_name, y_name = trans_feature_desc
            pre_y = state_dic[pre_y_name]
            y = state_dic[y_name]
            sum_k += omega_state_trans[pre_y, y] * state_trans_matrix[pre_y, y]
        # 在计算观察特征，格式为：{i: {t: {y_trans:{feature_desc: count}, X_trans:{feature_desc: count}}}}
        for state_feature_desc, _ in feature_dic_t['x_trans'].items():
            y_name, x_observe_name = state_feature_desc
            y = state_dic[y_name]
            x_observe = observe_dic[x_observe_name]
            sum_k += omega_observe_trans[x_observe, y] * observe_trans_matrix[x_observe, y]
        multipy_t *= exp(sum_k)

    # 计算Z_matrix, 元素为[t, pre_y, y]，Z_matrix的每个元素为每个时刻t，转移特征和状态特征的求和
    for t, feature_dic_t in feature_dic_i.items():
        for _, y in state_dic.items():
            if y == STARTING_INDEX:
                continue  # y如果为初始节点，则跳过，因为t时刻是从y1开始的。
            # 先算观察特征
            for state_feature_desc, _ in feature_dic_t['x_trans'].items():
                _, x_observe_name = state_feature_desc
                x_observe = observe_dic[x_observe_name]
                if t == 0:
                    Z_matrix[t, 0, y] += omega_observe_trans[x_observe, y] * observe_trans_matrix[x_observe, y]
                else:
                    Z_matrix[t, 1:, y] += omega_observe_trans[x_observe, y] * observe_trans_matrix[x_observe, y]
            # 再算转移特征
            if t == 0:
                Z_matrix[t, STARTING_INDEX, y] += omega_state_trans[STARTING_INDEX, y] * state_trans_matrix[
                    STARTING_INDEX, y]
            else:
                for _, pre_y in state_dic.items():
                    if pre_y > 0:
                        Z_matrix[t, pre_y, y] += omega_state_trans[pre_y, y] * state_trans_matrix[pre_y, y]

        Z_matrix[t, :, :] = exp(Z_matrix[t, :, :])
    # 所有Z矩阵相乘，得到每种y路径的统计
    for t, _ in feature_dic_i.items():
        if t == 0:
            Z_prob = Z_matrix[t, :, :]
        else:
            Z_prob = np.dot(Z_prob, Z_matrix[t, :, :])

    Z = np.sum(Z_prob)
    return multipy_t / Z


# # 在训练集上计算对数似然函数 l(w) = sum_i(li(w))
# def _log_likelihood(params, *args):
#     print("omega: {0}".format(sorted(params)[-100:]))
#     X, Y, state_dic, observe_dic, trans_feature, state_feature, feature_dic, \
#     state_trans_matrix, observe_trans_matrix, feature_num, squared_sigma = args
#     likelihood = 0
#     for i in range(len(X)):
#         likelihood += calc_log_likelihood_i(params, state_dic, observe_dic, trans_feature, state_feature, feature_dic,
#                                             state_trans_matrix, observe_trans_matrix)
#     # 增加惩罚因子, 用w的二阶范式
#     likelihood = likelihood - np.sum(np.dot(params, params)) / (squared_sigma * 2)
#     # 自然语言处理P208
#     gradients = feature_list - calc_feature_counts - params / squared_sigma
#     print("gradients: {0}".format(sorted(gradients)[-100:]))
#     global GRADIENT
#     GRADIENT = gradients
#
#     global SUB_ITERATION_NUM
#     sub_iteration_str = '    '
#     if SUB_ITERATION_NUM > 0:
#         sub_iteration_str = '(' + '{0:02d}'.format(SUB_ITERATION_NUM) + ')'
#     print('  ', '{0:03d}'.format(ITERATION_NUM), sub_iteration_str, ':', likelihood * -1)
#
#     SUB_ITERATION_NUM += 1
#
#     return likelihood * -1  # 迭代使得 -likelihood最小


# 对数似然函数计算
def _log_likelihood(params, *args):
    # print("omega: {0}".format(sorted(params)[-100:]))
    X, Y, state_dic, observe_dic, feature_dic, dataset_feature, feature_count, squared_sigma = args
    calc_feature_counts = np.zeros((len(feature_count)))
    state_num = len(state_dic)
    sum_logZ = 0
    for t, Xi_feature in dataset_feature.items():
        prob_M = build_probability_matrix(params, feature_dic, Xi_feature, len(state_dic))
        alpha, beta, Z_Xi, scaling_dic = forward_backword_calc(len(X[t]), prob_M, len(state_dic))
        # 计算sum(log(Z_Xi))
        sum_logZ += log(Z_Xi) + sum(log(scaling_coefficient) for _, scaling_coefficient in scaling_dic.items())

        for t, feature_t in Xi_feature.items():
            for (pre_y, y, x_observe_name), _ in feature_t.items():
                # 计算p(pre_y, y | Xi, t)
                if pre_y == -1:  # (11.32)
                    # if t in scaling_dic.keys():
                    #     prob = (alpha[t, y] * beta[t, y] * scaling_dic[t]) / Z_Xi
                    # else:
                    #     prob = (alpha[t, y] * beta[t, y]) / Z_Xi  # Refer to 统计学习方法 P225 (11.32)
                    # alpha[t], beta[t] 不会同时有scaling系数，与分母Z_xi的scaling系数抵消，所以不用乘scaling系数
                    prob = (alpha[t, y] * beta[t, y]) / Z_Xi  # Refer to 统计学习方法 P225 (11.32)
                else:  # (11.33)
                    if t == 0:
                        if pre_y is not STARTING_INDEX:
                            continue
                        else:
                            prob = (prob_M[t][STARTING_INDEX, y] * beta[t, y]) / Z_Xi
                    else:
                        if pre_y is STARTING_INDEX or y is STARTING_INDEX:
                            continue
                        else:  # Refer to 统计学习方法  P226 (11.33)
                            # alpha[t-1], beta[t] 会同时有scaling系数，与分母Z_xi的scaling系数抵消一个，还需要乘一个scaling系数
                            if t in scaling_dic.keys():
                                prob = ((alpha[t - 1, pre_y] * prob_M[t][pre_y, y] * beta[t, y]) / Z_Xi) * \
                                       scaling_dic[t]
                            else:
                                prob = (alpha[t - 1, pre_y] * prob_M[t][pre_y, y] * beta[t, y]) / Z_Xi
                feature_id = feature_dic[(pre_y, y, x_observe_name)]
                calc_feature_counts[feature_id] += prob

    # 概率学习方法P227 (11.37),自然语言处理P207 (6.7)
    likelihood = np.dot(feature_count, params) - sum_logZ - np.sum(np.dot(params, params)) / (squared_sigma * 2)
    # 自然语言处理P208
    gradients = feature_count - calc_feature_counts - params / squared_sigma
    print("gradients: max:{0}, min:{1}, avg:{2}".format(np.max(gradients), np.min(gradients), np.average(gradients)))
    global GRADIENT
    GRADIENT = gradients

    global SUB_ITERATION_NUM
    sub_iteration_str = '    '
    if SUB_ITERATION_NUM > 0:
        sub_iteration_str = '(' + '{0:02d}'.format(SUB_ITERATION_NUM) + ')'
    print('  ', '{0:03d}'.format(ITERATION_NUM), sub_iteration_str, ':', likelihood * -1)

    SUB_ITERATION_NUM += 1

    return likelihood * -1  # 迭代使得 -likelihood最小


def _gradient(params, *args):
    return GRADIENT * -1


def _callback(params):
    global ITERATION_NUM
    global SUB_ITERATION_NUM
    global TOTAL_SUB_ITERATIONS
    ITERATION_NUM += 1
    TOTAL_SUB_ITERATIONS += SUB_ITERATION_NUM
    SUB_ITERATION_NUM = 0


class CRF(object):

    def __init__(self, model_file='./model/crf_model.pkl', dic_file='./dictionary/dic_pos.txt', Task='CUT'):
        # 分词+词性标注模型词典
        self.dic_file = dic_file

        # CRF模型文件
        self.model_file = model_file

        # 参数加载，用于判断是否需要加载模型文件
        self.load_model = False

        # 区分不同的训练任务：CUT, POS, NER
        self.Task = Task

        # 读取并初始化字典
        self.dictionary = dic.dictionary(dic_file, Task)

        # CRF模型参数
        # 特征对象
        self.feature = Feature(Task)
        # CUT任务中，训练得出的omega权重向量，与CUT_feature_counter同纬度：M(X) = sum(omega * feature)
        self.CUT_omega = []
        # POS任务中，训练得出的omega权重向量，与POS_feature_counter同纬度：M(X) = sum(omega * feature)
        self.POS_omega = []
        # 平方sigma参数
        self.squared_sigma = 10

        # 是否生产模式
        self.PROD = False

        # 加载已训练好的CRF模型，如果没有取到，则重新训练
        try:
            if not os.path.exists(self.model_file):
                raise FileNotFoundError
            with open(self.model_file, 'rb') as f:
                self.feature = pickle.load(f)  # 特征对象
                self.CUT_omega = pickle.load(f)  # CUT分词权重向量
                self.POS_omega = pickle.load(f)  # POS词性标注权重向量
                print("load CRF model successful!")
                if len(self.CUT_omega.items()) == 0:
                    raise FileNotFoundError
        except EOFError:
            print("Can't find model file. Need train!")
            self.model_train()
        except FileNotFoundError:
            self.model_train()

    def model_train(self):
        if self.Task == 'CUT':
            X = self.dictionary.Z
            Y = self.dictionary.CUT_state
        elif self.Task == 'POS':
            # X = self.dictionary.Words
            X = self.dictionary.Z
            Y = self.dictionary.POS_state
        # 使用观测序列X，状态序列Y，构建特征空间
        self.feature.build_feature_dic(X, Y, self.Task)

        # # 通过特征空间，构建Y转移矩阵，作为训练数据
        # self.feature.build_state_trans_matrix()
        #
        # # 通过特征空间，构建X-Y的状态矩阵，作为训练数据
        # self.feature.build_observe_trans_matrix()

        state_dic = self.feature.y_state_dic
        observe_dic = self.feature.x_observe_dic
        feature_dic = self.feature.feature_dic
        dataset_feature = self.feature.dataset_feature
        feature_count = np.zeros((len(self.feature.feature_count)))
        for feature_id, count in self.feature.feature_count.items():
            feature_count[feature_id] = count

        print("* Number of states: {0}".format(len(state_dic)))
        print("* Number of observes: {0}".format(len(observe_dic)))
        print("* Number of features: {0}".format(len(feature_dic)))

        # 通过L-BFGS进行CRF模型参数训练，主要训练omega：每个feature的权重，使得训练集中P(Y|X)最大
        print('* Squared sigma:', self.squared_sigma)
        print('* L-BGFS开始训练')
        print('   ========================')
        print('   迭代方法: likelihood')
        print('   ------------------------')
        omega, log_likelihood, information = \
            fmin_l_bfgs_b(func=_log_likelihood, fprime=_gradient,
                          x0=np.zeros(len(feature_count)),
                          args=(X, Y, state_dic, observe_dic, feature_dic, dataset_feature, feature_count,
                                self.squared_sigma),
                          callback=_callback)
        print('   ========================')
        print('   (iter: iteration, sit: sub iteration)')
        print('* Training has been finished with %d iterations' % information['nit'])

        if information['warnflag'] != 0:
            print('* Warning (code: %d)' % information['warnflag'])
            if 'task' in information.keys():
                print('* Reason: %s' % (information['task']))
        print('* Likelihood: %s' % str(log_likelihood))

        if self.Task == 'CUT':
            self.CUT_omega = omega
        elif self.Task == 'POS':
            self.POS_omega = omega

        # 保存训练好的模型参数
        if self.PROD:
            self.save_model()

    def save_model(self):
        with open(self.model_file, 'wb') as f:
            pickle.dump(self.feature, f)  # 特征对象
            pickle.dump(self.y_trans_feature_all, f)  # Y状态转移数据统计
            pickle.dump(self.CUT_omega, f)  # 分词feature权重向量
            pickle.dump(self.POS_omega, f)  # POS词性feature权重向量


class Feature(object):
    def __init__(self, feature_define_file='./dictionary/crf_feature.txt',
                 feature_model_file='./model/crf_feature_model.pkl', Task='POS'):
        self.feature_define_file = feature_define_file
        self.feature_model_file = feature_model_file
        self.save_feature = False
        # 任务类型
        self.Task = Task
        # 特征模板
        self.feature_templates = list()
        # y状态序列的索引字典，格式为：{y_state_name, y_state_id}
        self.y_state_dic = {STARTING_INDEX, STARTING_STATE}
        # x观察序列的索引字典，格式为：{x_observe_id, x_observe_name}
        self.x_observe_dic = dict()
        # 全训练集上的特征字典，格式为：{feature_desc: feature_id}
        # feature_desc的格式为：(pre_y, y, x_observe_id) pre_y = -1 为观察序列的状态特征，pre_y <> -1 为状态序列的转移特征
        self.feature_dic = dict()
        # by训练样本结构的特征字典
        # 格式为：{i: {t: {feature_desc: count_by_dataset}}}
        # feature_desc的格式为：(pre_y, y, x_observe_id) pre_y = -1 为观察序列的状态特征，pre_y <> -1 为状态序列的转移特征
        self.dataset_feature = dict()
        # 全训练集的特征数量统计，格式为：{feature_id, count}
        self.feature_count = dict()

        # 预加载外部定义的feature文件
        if os.path.exists(self.feature_define_file):
            with open(self.feature_define_file, 'rb') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    else:
                        self.feature_templates.append(line)
        else:
            # 如果没有外部定义的feature文件，则采用内部默认定义的feature模板文件
            self.feature_templates = list()
            # U = Unigram, B = Bigram, T = Trigram
            # 在NLP的序列分析里，过去的信息的重要性始终大于未来的。
            self.feature_templates.append('U[0]')
            self.feature_templates.append('U[+1]')
            self.feature_templates.append('U[-1]')
            self.feature_templates.append('B[0]')
            self.feature_templates.append('B[+1]')
            self.feature_templates.append('B[-1]')
            self.feature_templates.append('T[-1]')
            self.feature_templates.append('T[-2]')

        # 基于训练集生成的特征字典, 并统计每个特征集的数量
        # 预加载已经训练好的feature model文件
        if os.path.exists(self.feature_model_file):
            with open(self.feature_model_file, 'rb') as f:
                self.y_state_dic = pickle.load(f)
                self.x_observe_dic = pickle.load(f)
                self.feature_dic = pickle.load(f)
                self.dataset_feature = pickle.load(f)
                self.feature_count = pickle.load(f)
                print("load feature model file successful!")

    def match_feature(self, feature_template, Xi, t):
        if feature_template == 'U[0]':
            return '{0}:{1}'.format(feature_template, Xi[t])
        elif feature_template == 'U[+1]':
            if t < len(Xi) - 1:
                return '{0}:{1}'.format(feature_template, Xi[t + 1])
        elif feature_template == 'U[+2]':
            if t < len(Xi) - 2:
                return '{0}:{1}'.format(feature_template, Xi[t + 2])
        elif feature_template == 'U[-1]':
            if t > 0:
                return '{0}:{1}'.format(feature_template, Xi[t - 1])
        elif feature_template == 'U[-2]':
            if t > 1:
                return '{0}:{1}'.format(feature_template, Xi[t - 2])
        elif feature_template == 'B[0]':
            if t < len(Xi) - 1:
                return '{0}:{1},{2}'.format(feature_template, Xi[t], Xi[t + 1])
        elif feature_template == 'B[+1]':
            if t < len(Xi) - 2:
                return '{0}:{1},{2}'.format(feature_template, Xi[t + 1], Xi[t + 2])
        elif feature_template == 'B[-1]':
            if t > 0:
                return '{0}:{1},{2}'.format(feature_template, Xi[t - 1], Xi[t])
        elif feature_template == 'B[-2]':
            if t > 1:
                return '{0}:{1},{2}'.format(feature_template, Xi[t - 2], Xi[t - 1])
        elif feature_template == 'T[-1]':
            if 0 < t < len(Xi) - 1:
                return '{0}:{1},{2},{3}'.format(feature_template, Xi[t - 1], Xi[t], Xi[t + 1])
        elif feature_template == 'T[-2]':
            if t > 1:
                return '{0}:{1},{2},{3}'.format(feature_template, Xi[t - 2], Xi[t - 1], Xi[t])

    def scan_state_feature(self, Xi, t):
        feature_list = []
        for feature_template in self.feature_templates:
            feature = self.match_feature(feature_template, Xi, t)
            if feature is not None:
                feature_list.append(feature)
        return feature_list

    # 构建特征字典，Task为任务类型，目前支持CUT和POS
    # 此方法只针对X，Y有相同结构的线性链CRF
    # 对CUT分词任务，X为字list，Y为{B,M,E,S} list
    # 对POS分词任务，X为词list，Y为词性list
    # 脚标的含义说明：i：训练集脚标(0,m)；t：时刻脚标(0, len(X[i]))；k：特征集脚标
    def build_feature_dic(self, X, Y, Task='POS'):
        # y状态序列的索引字典，格式为：{y_state_name: y_state_id}
        y_state_dic = {STARTING_STATE: STARTING_INDEX}
        # x观察序列的索引字典，格式为：{x_observe_name: x_observe_id}
        x_observe_dic = dict()
        # 全训练集上的特征字典，格式为：{feature_desc: feature_id}
        # feature_desc的格式为：(pre_y, y, x_observe_id) pre_y = -1 为观察序列的状态特征，pre_y <> -1 为状态序列的转移特征
        feature_dic = dict()
        # by训练样本结构的特征字典
        # 格式为：{i: {t: {feature_desc: count_by_dataset}}}
        # feature_desc的格式为：(pre_y, y, x_observe_id) pre_y = -1 为观察序列的状态特征，pre_y <> -1 为状态序列的转移特征
        dataset_feature = dict()
        # 全训练集的特征数量统计，格式为：{feature_id, count}
        feature_count = dict()
        pre_y = STARTING_INDEX
        for i in range(len(X)):
            dataset_feature[i] = dict()
            for t in range(len(X[i])):
                dataset_feature[i][t] = dict()
                if Y[i][t] not in y_state_dic.keys():
                    y_state_dic[Y[i][t]] = len(y_state_dic)  # 更新y状态序列的索引字典
                y = y_state_dic[Y[i][t]]
                observe_feature_list = self.scan_state_feature(X[i], t)
                for x_observe_name in observe_feature_list:
                    if x_observe_name not in x_observe_dic.keys():
                        x_observe_dic[x_observe_name] = len(x_observe_dic)  # 更新x观察序列的索引字典
                    x_observe_id = x_observe_dic[x_observe_name]
                    # 先计算状态序列的转移特征
                    if (pre_y, y, x_observe_id) in feature_dic.keys():
                        feature_count[feature_dic[(pre_y, y, x_observe_id)]] += 1
                        if (pre_y, y, x_observe_id) in dataset_feature[i][t].keys():
                            dataset_feature[i][t][(pre_y, y, x_observe_id)] += 1
                        else:
                            dataset_feature[i][t][(pre_y, y, x_observe_id)] = 1
                    else:
                        new_feature_id = len(feature_count)
                        feature_count[new_feature_id] = 1
                        feature_dic[(pre_y, y, x_observe_id)] = new_feature_id
                        dataset_feature[i][t][(pre_y, y, x_observe_id)] = 1
                    # 再计算观察序列的状态特征
                    if (-1, y, x_observe_id) in feature_dic.keys():
                        feature_count[feature_dic[(-1, y, x_observe_id)]] += 1
                        if (-1, y, x_observe_id) in dataset_feature[i][t].keys():
                            dataset_feature[i][t][(-1, y, x_observe_id)] += 1
                        else:
                            dataset_feature[i][t][(-1, y, x_observe_id)] = 1
                    else:
                        new_feature_id = len(feature_count)
                        feature_count[new_feature_id] = 1
                        feature_dic[(-1, y, x_observe_id)] = new_feature_id
                        dataset_feature[i][t][(-1, y, x_observe_id)] = 1
                pre_y = y

        self.y_state_dic = y_state_dic
        self.x_observe_dic = x_observe_dic
        self.feature_dic = feature_dic
        self.dataset_feature = dataset_feature
        self.feature_count = feature_count

        # 是否为训练模式，需要保存模型
        if self.save_feature:
            self.save_feature_model()

    # def build_state_trans_matrix(self):
    #     state_trans_matrix = np.zeros((len(self.y_state_dic), len(self.y_state_dic)))
    #     # y状态序列的索引字典，格式为：{y_state_name, y_state_id}
    #     # 在全训练集下的转移特征，格式为：{feature_desc: (id, count)}
    #     # feature_desc的格式为：(y_state_name[t-1], y_state_name[t])
    #     for feature_desc, (_, count) in self.trans_feature.items():
    #         pre_y_name, y_name = feature_desc
    #         pre_y = self.y_state_dic[pre_y_name]
    #         y = self.y_state_dic[y_name]
    #         state_trans_matrix[pre_y, y] = count
    #     self.state_trans_matrix = state_trans_matrix
    #
    # def build_observe_trans_matrix(self):
    #     observe_trans_matrix = np.zeros(())
    #     # x观察序列的索引字典，格式为：{x_observe_name, x_observe_id}
    #     # 在全训练集下的状态特征，格式为：{feature_desc: (id, count)}
    #     # feature_desc的格式为：(y_state_name[t], x_observe_name[t])
    #     for feature_desc, (_, count) in self.state_feature.items():
    #         y_name, x_observe_name = feature_desc
    #         y_state = self.y_state_dic[y_name]
    #         x_observe = self.x_observe_dic[x_observe_name]
    #         observe_trans_matrix[x_observe, y_state] = count
    #     self.observe_trans_matrix = observe_trans_matrix

    def save_feature_model(self):
        with open(self.feature_model_file, 'wb') as file:
            pickle.dump(self.y_state_dic, file)
            pickle.dump(self.x_observe_dic, file)
            pickle.dump(self.feature_dic, file)
            pickle.dump(self.dataset_feature, file)
            pickle.dump(self.feature_count, file)


if __name__ == '__main__':
    # type: Normal普通分词, POS词性分词, NER命名实体识别
    crf = CRF(model_file='./model/crf_model.pkl', dic_file='./dictionary/dic_pos_simple.txt', Task='POS')
    # res = list(crf.cut("我想学习计算机编程"))
    # print(res)
    #
    # cont = True
    # while cont:
    #     text = str(input("请输入你想分词的句子： "))
    #     if text == "exit":
    #         cont = False
    #     else:
    #         res = crf.cut(text)
    #         print(str(list(res)))
