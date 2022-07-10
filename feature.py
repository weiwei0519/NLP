# coding=utf-8
# 特征函数构造

import os
import sys
import pickle
from collections import Counter
import numpy as np

STARTING_STATE = '*'  # STARTING_STATE
STARTING_INDEX = 0

class Feature(object):
    def __init__(self, feature_define_file='./dictionary/crf_feature.txt',
                 feature_model_file='./model/crf_feature_model.pkl'):
        self.feature_define_file = feature_define_file
        self.feature_model_file = feature_model_file
        self.save_feature = False
        # 分词状态字典
        self.CUT_state_dic = {STARTING_STATE: STARTING_INDEX}
        # POS状态字典
        self.POS_state_dic = {STARTING_STATE: STARTING_INDEX}
        # 特征模板
        self.feature_templates = list()
        # 分词特征字典
        self.CUT_feature_dic = dict()
        # POS标注特征字典
        self.POS_feature_dic = dict()
        # 分词特征总数
        self.CUT_feature_num = 0
        # POS标注特征总数
        self.POS_feature_num = 0
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
            self.feature_templates.append('U[+2]')
            self.feature_templates.append('U[-1]')
            self.feature_templates.append('U[-2]')
            self.feature_templates.append('B[0]')
            self.feature_templates.append('B[+1]')
            self.feature_templates.append('B[-1]')
            self.feature_templates.append('B[-2]')
            self.feature_templates.append('T[-1]')
            self.feature_templates.append('T[-2]')

        # 基于训练集生成的特征字典, 并统计每个特征集的数量
        # 预加载已经训练好的feature model文件
        if os.path.exists(self.feature_model_file):
            with open(self.feature_model_file, 'rb') as f:
                self.CUT_state_dic = pickle.load(f)
                self.POS_state_dic = pickle.load(f)
                self.CUT_feature_dic = pickle.load(f)
                self.POS_feature_dic = pickle.load(f)
                self.CUT_feature_counter = pickle.load(f)
                self.POS_feature_counter = pickle.load(f)
                self.CUT_feature_num = pickle.load(f)
                self.POS_feature_num = pickle.load(f)
                print("load feature model file successful!")
        else:
            self.CUT_state_dic = dict()
            self.POS_state_dic = dict()
            self.CUT_feature_dic = dict()
            self.POS_feature_dic = dict()
            self.CUT_feature_counter = Counter()
            self.POS_feature_counter = Counter()
            self.CUT_feature_num = 0
            self.POS_feature_num = 0

    def match_feature(self, feature_template, X, i):
        if feature_template == 'U[0]':
            return '{0}:{1}'.format(feature_template, X[i])
        elif feature_template == 'U[+1]':
            if i < len(X) - 1:
                return '{0}:{1}'.format(feature_template, X[i + 1])
        elif feature_template == 'U[+2]':
            if i < len(X) - 2:
                return '{0}:{1}'.format(feature_template, X[i + 2])
        elif feature_template == 'U[-1]':
            if i > 0:
                return '{0}:{1}'.format(feature_template, X[i - 1])
        elif feature_template == 'U[-2]':
            if i > 1:
                return '{0}:{1}'.format(feature_template, X[i - 2])
        elif feature_template == 'B[0]':
            if i < len(X) - 1:
                return '{0}:{1},{2}'.format(feature_template, X[i], X[i + 1])
        elif feature_template == 'B[+1]':
            if i < len(X) - 2:
                return '{0}:{1},{2}'.format(feature_template, X[i + 1], X[i + 2])
        elif feature_template == 'B[-1]':
            if i > 0:
                return '{0}:{1},{2}'.format(feature_template, X[i - 1], X[i])
        elif feature_template == 'B[-2]':
            if i > 1:
                return '{0}:{1},{2}'.format(feature_template, X[i - 2], X[i - 1])
        elif feature_template == 'T[-1]':
            if 0 < i < len(X) - 1:
                return '{0}:{1},{2},{3}'.format(feature_template, X[i - 1], X[i], X[i + 1])
        elif feature_template == 'T[-2]':
            if i > 1:
                return '{0}:{1},{2},{3}'.format(feature_template, X[i - 2], X[i - 1], X[i])

    def scan_features(self, Xi, t):
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
    def build_feature_dic(self, X, Y, Task='CUT'):
        state_dic = {STARTING_STATE: STARTING_INDEX}
        feature_dic = dict()
        feature_counter = Counter()
        feature_num = 0
        pre_y = STARTING_INDEX
        for m in range(len(X)):
            for i in range(len(X[m])):
                if Y[m][i] not in state_dic.keys():
                    state_dic[Y[m][i]] = len(state_dic)
                y = state_dic[Y[m][i]]
                feature_list = self.scan_features(X[m], i)
                for feature_name in feature_list:
                    if feature_name in feature_dic.keys():
                        if (pre_y, y) in feature_dic[feature_name].keys():
                            feature_counter[feature_dic[feature_name][(pre_y, y)]] += 1
                        else:
                            feature_id = len(feature_counter)
                            feature_dic[feature_name][(pre_y, y)] = feature_id
                            feature_counter[feature_id] = 1
                            feature_num += 1
                        if (-1, y) in feature_dic[feature_name].keys():
                            feature_counter[feature_dic[feature_name][(-1, y)]] += 1
                        else:
                            feature_id = len(feature_counter)
                            feature_dic[feature_name][(-1, y)] = feature_id
                            feature_counter[feature_id] = 1
                            feature_num += 1
                    else:
                        feature_dic[feature_name] = dict()
                        feature_id = len(feature_counter)
                        feature_dic[feature_name][(pre_y, y)] = feature_id
                        feature_counter[feature_id] = 1
                        feature_num += 1
                        feature_id = len(feature_counter)
                        feature_dic[feature_name][(-1, y)] = feature_id
                        feature_counter[feature_id] = 1
                        feature_num += 1
                pre_y = y
        if Task == 'CUT':
            self.CUT_state_dic = state_dic
            self.CUT_feature_dic = feature_dic
            self.CUT_feature_num = feature_num
            self.CUT_feature_counter = feature_counter
        elif Task == 'POS':
            self.POS_state_dic = state_dic
            self.POS_feature_dic = feature_dic
            self.POS_feature_num = feature_num
            self.POS_feature_counter = feature_counter
        if self.save_feature:
            self.save_feature_model()

    def get_feature_counts_array(self, Task='CUT'):
        if Task == 'CUT':
            feature_counts = np.ndarray((self.CUT_feature_num,))
            for feature_id, counts in self.CUT_feature_counter.items():
                feature_counts[feature_id] = counts
        elif Task == 'POS':
            feature_counts = np.ndarray((self.POS_feature_num,))
            for feature_id, counts in self.POS_feature_counter.items():
                feature_counts[feature_id] = counts
        return feature_counts

    # def save_feature_model(self):
    #     with open(self.feature_model_file, 'wb') as file:
    #         pickle.dump(self.feature_dic, file)
    #         pickle.dump(self.feature_counts, file)