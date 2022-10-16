# coding=UTF-8
# 条件随机场概率模型进行序列标注

"""
Laon-CRF
    : Easy-to-use Linear Chain Conditional Random Fields
Author: Seong-Jin Kim
Modified by: Wei Wei
License: MIT License
Version: 0.0
Email: lancifollia@gmail.com
Created: May 13, 2015
Modified: May 23, 2022
Copyright (c) 2015 Seong-Jin Kim
"""

from crf_feature import FeatureSet, STARTING_LABEL_INDEX, CUT_feature_func

from math import exp, log
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import dictionary as dic
import random
import time
import json
import datetime

from collections import Counter

SCALING_THRESHOLD = 1e250

ITERATION_NUM = 0
SUB_ITERATION_NUM = 0
TOTAL_SUB_ITERATIONS = 0
GRADIENT = None


def _callback(params):
    global ITERATION_NUM
    global SUB_ITERATION_NUM
    global TOTAL_SUB_ITERATIONS
    ITERATION_NUM += 1
    TOTAL_SUB_ITERATIONS += SUB_ITERATION_NUM
    SUB_ITERATION_NUM = 0


def _generate_potential_table(params, num_labels, feature_set, X_features, inference=True):
    """
    Generates a potential table using given observations.
    * potential_table[t][prev_y, y]
        := exp(inner_product(params, feature_vector(prev_y, y, X, t)))
        (where 0 <= t < len(X))
    """
    tables = list()
    for t in range(len(X_features)):
        table = np.zeros((num_labels, num_labels))  # 概率矩阵的元素的含义是：从状态（行）转移到状态（列）
        if inference:
            for (prev_y, y), score in feature_set.calc_inner_products(params, X_features, t):
                if prev_y == -1:
                    table[:, y] += score
                else:
                    table[prev_y, y] += score
        else:
            for (prev_y, y), feature_ids in X_features[t]:
                score = sum(params[fid] for fid in feature_ids)  # 注意此处计算的是权重x特征函数，不是每个特征出现的次数
                if prev_y == -1:
                    table[:, y] += score    # CRF的矩阵形式（11.23），概率矩阵的每个元素为t时刻，每个状态转移背后：特征权重x特征函数的求和
                else:
                    table[prev_y, y] += score
        table = np.exp(table)
        if t == 0:
            table[STARTING_LABEL_INDEX + 1:] = 0
        else:
            table[:, STARTING_LABEL_INDEX] = 0
            table[STARTING_LABEL_INDEX, :] = 0
        tables.append(table)

    return tables


def _forward_backward(num_labels, time_length, potential_table):
    """
    Calculates alpha(forward terms), beta(backward terms), and Z(instance-specific normalization factor)
        with a scaling method(suggested by Rabiner, 1989).
    * Reference:
        - 1989, Lawrence R. Rabiner, A Tutorial on Hidden Markov Models and Selected Applications
        in Speech Recognition
    """
    alpha = np.zeros((time_length, num_labels))
    scaling_dic = dict()
    t = 0
    for label_id in range(num_labels):
        alpha[t, label_id] = potential_table[t][STARTING_LABEL_INDEX, label_id]
    # alpha[0, :] = potential_table[0][STARTING_LABEL_INDEX, :]  # slow
    t = 1
    while t < time_length:
        scaling_time = None
        scaling_coefficient = None
        overflow_occured = False
        label_id = 1
        while label_id < num_labels:
            alpha[t, label_id] = np.dot(alpha[t - 1, :], potential_table[t][:, label_id])
            if alpha[t, label_id] > SCALING_THRESHOLD:
                if overflow_occured:
                    print('******** Consecutive overflow ********')
                    raise BaseException()
                overflow_occured = True
                scaling_time = t - 1
                scaling_coefficient = SCALING_THRESHOLD
                scaling_dic[scaling_time] = scaling_coefficient
                break
            else:
                label_id += 1
        if overflow_occured:
            alpha[t - 1] /= scaling_coefficient
            alpha[t] = 0
        else:
            t += 1

    beta = np.zeros((time_length, num_labels))
    t = time_length - 1
    for label_id in range(num_labels):
        beta[t, label_id] = 1.0
    # beta[time_length - 1, :] = 1.0     # slow
    for t in range(time_length - 2, -1, -1):
        for label_id in range(1, num_labels):
            beta[t, label_id] = np.dot(beta[t + 1, :], potential_table[t + 1][label_id, :])
        if t in scaling_dic.keys():
            beta[t] /= scaling_dic[t]

    Z = sum(alpha[time_length - 1])

    return alpha, beta, Z, scaling_dic


def _calc_path_score(potential_table, scaling_dic, Y, label_dic):
    score = 1.0
    prev_y = STARTING_LABEL_INDEX
    for t in range(len(Y)):
        y = label_dic[Y[t]]
        score *= potential_table[prev_y, y, t]
        if t in scaling_dic.keys():
            score = score / scaling_dic[t]
        prev_y = y
    return score


def _log_likelihood(params, *args):
    """
    Calculate likelihood and gradient
    """
    training_data, feature_set, training_feature_data, empirical_counts, label_dic, squared_sigma = args
    expected_counts = np.zeros(len(feature_set))
    # print("params: {0}".format(sorted(params)[-100:]))
    total_logZ = 0
    for X_features in training_feature_data:
        potential_table = _generate_potential_table(params, len(label_dic), feature_set,
                                                    X_features, inference=False)
        alpha, beta, Z, scaling_dic = _forward_backward(len(label_dic), len(X_features), potential_table)
        total_logZ += log(Z) + \
                      sum(log(scaling_coefficient) for _, scaling_coefficient in scaling_dic.items())
        for t in range(len(X_features)):
            potential = potential_table[t]
            for (prev_y, y), feature_ids in X_features[t]:
                # Adds p(prev_y, y | X, t)
                if prev_y == -1:
                    if t in scaling_dic.keys():
                        prob = (alpha[t, y] * beta[t, y] * scaling_dic[t]) / Z
                    else:
                        prob = (alpha[t, y] * beta[t, y]) / Z
                elif t == 0:
                    if prev_y is not STARTING_LABEL_INDEX:
                        continue
                    else:
                        prob = (potential[STARTING_LABEL_INDEX, y] * beta[t, y]) / Z
                else:
                    if prev_y is STARTING_LABEL_INDEX or y is STARTING_LABEL_INDEX:
                        continue
                    else:
                        prob = (alpha[t - 1, prev_y] * potential[prev_y, y] * beta[t, y]) / Z
                for fid in feature_ids:
                    expected_counts[fid] += prob

    likelihood = np.dot(empirical_counts, params) - total_logZ - \
                 np.sum(np.dot(params, params)) / (squared_sigma * 2)

    gradients = empirical_counts - expected_counts - params / squared_sigma
    print("gradients: max:{0}, min:{1}, avg:{2}".format(np.max(gradients), np.min(gradients), np.average(gradients)))
    global GRADIENT
    GRADIENT = gradients

    global SUB_ITERATION_NUM
    sub_iteration_str = '    '
    if SUB_ITERATION_NUM > 0:
        sub_iteration_str = '(' + '{0:02d}'.format(SUB_ITERATION_NUM) + ')'
    print('  ', '{0:03d}'.format(ITERATION_NUM), sub_iteration_str, ':', likelihood * -1)

    SUB_ITERATION_NUM += 1

    return likelihood * -1


def _gradient(params, *args):
    return GRADIENT * -1


class LinearChainCRF():
    """
    Linear-chain Conditional Random Field
    """

    training_data = None
    feature_set = None

    label_dic = None
    label_array = None
    num_labels = None

    params = None

    # For L-BFGS
    squared_sigma = 10.0

    def __init__(self):
        pass

    @staticmethod
    def read_conll_corpus(filename):
        """
        Read a corpus file with a format used in CoNLL.
        """
        dictionary = dic.dictionary(filename, 'POS')
        training_data = list()
        testing_data = list()
        X = dictionary.Z
        Y = dictionary.POS_state
        # 随机选取10%作为测试集
        testing_index = random.sample(list(range(len(X))), int(np.round(0.2 * len(X))))
        for i in range(len(X)):
            if i in testing_index:
                testing_data.append((X[i], Y[i]))
            else:
                training_data.append((X[i], Y[i]))

        # data_string_list = list(open(filename, encoding='utf-8'))
        #
        # element_size = 0
        # X = list()
        # Y = list()
        # for data_string in data_string_list:
        #     words = data_string.strip().split()
        #     if len(words) is 0:
        #         datasets.append((X, Y))
        #         X = list()
        #         Y = list()
        #     else:
        #         if element_size is 0:
        #             element_size = len(words)
        #         elif element_size is not len(words):
        #             return
        #         X.append(words[:-1])
        #         Y.append(words[-1])
        # if len(X) > 0:
        #     datasets.append((X, Y))

        return training_data, testing_data

    def _read_corpus(self, filename):
        return self.read_conll_corpus(filename)

    # 范围二维数组，每行是一个Xi的特征集，每个元素是X[i][t]的特征集，
    # 每个元素，去掉了对X通过特征模板匹配的特征，只保留状态变化特征(pre_y, y)和(-1, y)，以及在feature_dic中的feature id
    # 这个二维数组实际上表示了一个：状态转移特征(pre_y, y)和(-1, y)背后关联了哪些X_feature
    # 后续用于概率矩阵的计算，因为概率矩阵是t时刻，状态转移特征的统计
    def _get_training_feature_data(self):
        return [[self.feature_set.get_feature_list(X, t) for t in range(len(X))]
                for X, _ in self.training_data]

    def _estimate_parameters(self):
        """
        Estimates parameters using L-BFGS.
        * References:
            - R. H. Byrd, P. Lu and J. Nocedal. A Limited Memory Algorithm for Bound Constrained Optimization,
            (1995), SIAM Journal on Scientific and Statistical Computing, 16, 5, pp. 1190-1208.
            - C. Zhu, R. H. Byrd and J. Nocedal. L-BFGS-B: Algorithm 778: L-BFGS-B, FORTRAN routines for large
            scale bound constrained optimization (1997), ACM Transactions on Mathematical Software, 23, 4,
            pp. 550 - 560.
            - J.L. Morales and J. Nocedal. L-BFGS-B: Remark on Algorithm 778: L-BFGS-B, FORTRAN routines for
            large scale bound constrained optimization (2011), ACM Transactions on Mathematical Software, 38, 1.
        """
        training_feature_data = self._get_training_feature_data()
        print('* Squared sigma:', self.squared_sigma)
        print('* Start L-BGFS')
        print('   ========================')
        print('   iter(sit): likelihood')
        print('   ------------------------')
        self.params, log_likelihood, information = \
            fmin_l_bfgs_b(func=_log_likelihood, fprime=_gradient,
                          x0=np.zeros(len(self.feature_set)),
                          args=(self.training_data, self.feature_set, training_feature_data,
                                self.feature_set.get_empirical_counts(),
                                self.label_dic, self.squared_sigma),
                          callback=_callback)
        print('   ========================')
        print('   (iter: iteration, sit: sub iteration)')
        print('* Training has been finished with %d iterations' % information['nit'])

        if information['warnflag'] != 0:
            print('* Warning (code: %d)' % information['warnflag'])
            if 'task' in information.keys():
                print('* Reason: %s' % (information['task']))
        print('* Likelihood: %s' % str(log_likelihood))

    def train(self, corpus_filename, model_filename):
        """
        Estimates parameters using conjugate gradient methods.(L-BFGS-B used)
        """
        start_time = time.time()
        print('[%s] Start training' % datetime.datetime.now())

        # Read the training corpus
        print("* Reading training datasets ... \n", end="")
        self.training_data, self.testing_data = self._read_corpus(corpus_filename)
        print("Done")

        # Generate feature set from the corpus
        self.feature_set = FeatureSet(CUT_feature_func)
        self.feature_set.scan(self.training_data)
        self.label_dic, self.label_array = self.feature_set.get_labels()
        self.num_labels = len(self.label_array)
        print("* Number of labels: %d" % (self.num_labels - 1))
        print("* Number of features: %d" % len(self.feature_set))

        # Estimates parameters to maximize log-likelihood of the corpus.
        self._estimate_parameters()

        self.save_model(model_filename)

        elapsed_time = time.time() - start_time
        print('* Elapsed time: %f' % elapsed_time)
        print('* [%s] Training done' % datetime.datetime.now())

    def test(self, test_corpus_filename):
        if self.params is None:
            raise BaseException("You should load a model first!")

        if test_corpus_filename is None:
            test_data = self.testing_data
        else:
            test_data = self._read_corpus(test_corpus_filename)

        total_count = 0
        correct_count = 0
        for X, Y in test_data:
            Yprime = self.inference(X)
            for t in range(len(Y)):
                total_count += 1
                if Y[t] == Yprime[t]:
                    correct_count += 1

        print('Correct: %d' % correct_count)
        print('Total: %d' % total_count)
        print('Performance: %f' % (correct_count / total_count))

    def print_test_result(self, test_corpus_filename):
        test_data = self._read_corpus(test_corpus_filename)

        for X, Y in test_data:
            Yprime = self.inference(X)
            for t in range(len(X)):
                print('%s\t%s\t%s' % ('\t'.join(X[t]), Y[t], Yprime[t]))
            print()

    def inference(self, X):
        """
        Finds the best label sequence.
        """
        self.feature_set = FeatureSet(CUT_feature_func)
        potential_table = _generate_potential_table(self.params, self.num_labels,
                                                    self.feature_set, X, inference=True)
        Yprime = self.viterbi(X, potential_table)
        return Yprime

    def viterbi(self, X, potential_table):
        """
        The Viterbi algorithm with backpointers
        """
        time_length = len(X)
        max_table = np.zeros((time_length, self.num_labels))
        argmax_table = np.zeros((time_length, self.num_labels), dtype='int64')

        t = 0
        for label_id in range(self.num_labels):
            max_table[t, label_id] = potential_table[t][STARTING_LABEL_INDEX, label_id]
        for t in range(1, time_length):
            for label_id in range(1, self.num_labels):
                max_value = -float('inf')
                max_label_id = None
                for prev_label_id in range(1, self.num_labels):
                    value = max_table[t - 1, prev_label_id] * potential_table[t][prev_label_id, label_id]
                    if value > max_value:
                        max_value = value
                        max_label_id = prev_label_id
                max_table[t, label_id] = max_value
                argmax_table[t, label_id] = max_label_id

        sequence = list()
        next_label = max_table[time_length - 1].argmax()
        sequence.append(next_label)
        for t in range(time_length - 1, -1, -1):
            next_label = argmax_table[t, next_label]
            sequence.append(next_label)
        return [self.label_dic[label_id] for label_id in sequence[::-1][1:]]

    def save_model(self, model_filename):
        model = {"feature_dic": self.feature_set.serialize_feature_dic(),
                 "num_features": self.feature_set.num_features,
                 "labels": self.feature_set.label_array,
                 "params": list(self.params)}
        f = open(model_filename, 'w')
        json.dump(model, f, ensure_ascii=False, indent=2, separators=(',', ':'))
        f.close()
        import os
        print('* Trained CRF Model has been saved at "%s/%s"' % (os.getcwd(), model_filename))

    def load(self, corpus_filename, model_filename):
        f = open(model_filename)
        model = json.load(f)
        f.close()

        # Read the training corpus
        print("* Reading training datasets ... ", end="")
        self.training_data, self.testing_data = self._read_corpus(corpus_filename)
        print("Done")

        self.feature_set = FeatureSet()
        self.feature_set.load(model['feature_dic'], model['num_features'], model['labels'])
        self.label_dic, self.label_array = self.feature_set.get_labels()
        self.num_labels = len(self.label_array)
        self.params = np.asarray(model['params'])

        print('CRF model loaded')


# For testing
crf = LinearChainCRF()

# crf.train('datasets/simple_train.datasets', 'model/crf_bfgs.json')
crf.train('dictionary/dic_pos.txt', 'model/crf_bfgs.json')
crf.load('dictionary/dic_pos.txt', 'model/crf_bfgs.json')
crf.test()

# crf.train('datasets/chunking_2/train.txt', 'datasets/chunking_2/model_4.json')
# crf.load('datasets/chunking_2/model_4.json')
# crf.test('datasets/chunking_2/test.txt')
