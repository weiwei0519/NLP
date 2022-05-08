# coding=UTF-8
# 隐马尔科夫概率模型分词

import os
import pickle


class HMM(object):
    def __init__(self, model_file='./model/hmm_model.pkl', dic_file='./dictionary/dic_pos.txt', need_pos=True):

        # # 分词模型词典
        # self.cut_dic_file = './dictionary/dictionary.txt'
        #
        # # hmm分词模型，主要是用于存储算法模型的结果，是训练模式还是测试模式
        # self.cut_model_file = './model/hmm_cut_model.pkl'

        # 分词+词性标注模型词典
        self.dic_file = dic_file

        # hmm分词+词性标注模型，主要是用于存储算法模型的结果，是训练模式还是测试模式
        self.model_file = model_file

        # # 分词模型状态枚举值
        # self.cut_state_list = ['B', 'M', 'E', 'S']

        # 分词+词性标注模型状态枚举值
        self.state_list = {}

        # 参数加载，用于判断是否需要加载模型文件
        self.load_model = False

        # 区分是否需要附加词性
        self.need_pos = need_pos

        # HMM模型初始化参数
        # Pi，A，B概率
        # Pi：{'B_pos':0.0, 'M_pos':0.0, 'E_pos':0.0, 'S_pos':0.0}
        # B: # {'B_pos':{'字'：p}, 'M_pos':{'字'：p}, 'E_pos':{'字'：p}, 'S_pos':{'字'：p}}
        # A: # {'状态A':{'状态B'：p}}
        self.Pi_dic = {}
        self.B_dic = {}
        self.A_dic = {}
        # 加载已训练好的HMM模型，如果没有取到，则重新训练
        try:
            if not os.path.exists(self.model_file):
                raise FileNotFoundError
            with open(self.model_file, 'rb') as f:
                self.state_list = pickle.load(f)  # 状态序列集及统计数据
                self.A_dic = pickle.load(f)  # 状态转移概率
                self.B_dic = pickle.load(f)  # 发射概率概率
                self.Pi_dic = pickle.load(f)  # 状态的初始概率
                print("load HMM model successful!")
                if len(self.state_list.items()) == 0:
                    raise FileNotFoundError
        except EOFError:
            print("Can't find model file. Need train!")
            self.model_train()
        except FileNotFoundError:
            self.model_train()

    # 给输入word计算序列，并附上pos
    @staticmethod
    def makeLabel(word, pos):
        labels = []
        if len(pos) > 0:
            if len(word) == 1:
                labels.append('S_' + pos)
            else:
                labels += ['B_' + pos] + ['M_' + pos] * (len(word) - 2) + ['E_' + pos]
        else:
            if len(word) == 1:
                labels.append('S')
            else:
                labels += ['B'] + ['M'] * (len(word) - 2) + ['E']
        return labels

    # 通过给定的分词+词性标注语料，训练语料，计算转移概率、发射概率以及初始概率
    def model_train(self):
        print("start HMM cut model training")
        line_num = 0
        with open(self.dic_file, encoding='utf8') as f:  # 读取字典文件
            for line in f:
                line_num += 1
                # print("line_num: {0}".format(line_num))
                line = line.strip()  # 移除字符串头尾指定的字符（默认为空格或换行符）或字符序列
                if not line: continue
                words_org = line.split(' ')  # 词列表
                z = []
                labels = []
                words = []
                for word in words_org:  # 国务院/nt  [西藏/ns 自治区/n 政府/n]nt
                    # 首先去除命名实体的标志[]
                    if word.find('[') > 0:
                        word = word[word.find('[') + 1:-1]
                    if word.find(']') > 0:
                        word = word[:word.find(']')]
                    # 区分字典是否包含pos
                    if self.need_pos:
                        pos = word[word.find('/') + 1:]
                        word = word[:word.find('/')]
                    else:
                        pos = ''
                    z.extend(i for i in word)
                    words.append(word)
                    labels.extend(self.makeLabel(word, pos))
                assert len(z) == len(labels)
                for i in range(len(z)):
                    # 添加状态序列及计数
                    self.state_list[labels[i]] = self.state_list.get(labels[i], 0) + 1

                    # 发射概率计数
                    if labels[i] not in self.B_dic:
                        self.B_dic[labels[i]] = {}
                        self.B_dic[labels[i]][z[i]] = 1
                    elif labels[i] in self.B_dic and z[i] not in self.B_dic[labels[i]]:
                        self.B_dic[labels[i]][z[i]] = 1
                    else:
                        self.B_dic[labels[i]][z[i]] += 1

                    if i == 0:
                        # 初始概率计数，只计算一句话的首字
                        self.Pi_dic[labels[i]] = self.Pi_dic.get(labels[i], 0) + 1
                    else:
                        # 状态转移计数，从句首第二字开始计数
                        # 数据结构调整一下：{状态A：{状态B：p}}，这样好纵向扩展，减少全局调整
                        if labels[i - 1] not in self.A_dic:
                            self.A_dic[labels[i - 1]] = {}
                            self.A_dic[labels[i - 1]][labels[i]] = 1
                        elif labels[i] not in self.A_dic[labels[i - 1]]:
                            self.A_dic[labels[i - 1]][labels[i]] = 1
                        else:
                            self.A_dic[labels[i - 1]][labels[i]] += 1

            # 计算Pi，A，B的概率
            # Pi：{'B_pos':0.0, 'M_pos':0.0, 'E_pos':0.0, 'S_pos':0.0}
            # B: # {'B_pos':{'字'：p}, 'M_pos':{'字'：p}, 'E_pos':{'字'：p}, 'S_pos':{'字'：p}}
            # A: # {'状态A':{'状态B'：p}}
            self.Pi_dic = {label: self.Pi_dic.get(label, 0) / sum(self.Pi_dic.values()) for label in
                           self.state_list.keys()}
            # B_dic: # {'B_pos':{'字'：p}, 'M_pos':{'字'：p}, 'E_pos':{'字'：p}, 'S_pos':{'字'：p}}
            self.B_dic = {label: {zi: count / self.state_list[label] for zi, count in v.items()}
                          for label, v in self.B_dic.items()}
            # 对于B_dic中不存在的label，进行空赋值
            self.B_dic.update({label: {} for label, v in self.state_list.items() if label not in self.B_dic})
            # A: # {'状态A':{'状态B'：p}}
            self.A_dic = {label_A: {label_B: count / self.state_list[label_A] for label_B, count in v.items()}
                          for label_A, v in self.A_dic.items()}
            # 对于A_dic中不存在的label，进行空赋值
            self.A_dic.update({label_A: {label_B: 0.0 for label_B, v in self.state_list.items()} for label_A, v in
                               self.state_list.items() if label_A not in self.A_dic})
            for label_A in self.A_dic.keys():
                self.A_dic[label_A].update({label_B: 0.0 for label_B, v_B in self.state_list.items() if
                                            label_B not in self.A_dic[label_A]})

        with open(self.model_file, 'wb') as file:
            pickle.dump(self.state_list, file)
            pickle.dump(self.A_dic, file)
            pickle.dump(self.B_dic, file)
            pickle.dump(self.Pi_dic, file)
        print("end HMM cut model training")

    def cut(self, text):
        prob, label_list = self.viterbi(text, self.state_list, self.Pi_dic, self.A_dic, self.B_dic)
        if self.need_pos:
            pos_list = [label[label.find('_') + 1:] for label in label_list]
            label_list = [label[:label.find('_')] for label in label_list]
        else:
            pos_list = ['' for label in label_list]
        begin, next = 0, 0
        for i, char in enumerate(text):
            label = label_list[i]
            if label == 'B':
                begin = i
            elif label == 'E':
                yield text[begin: i + 1] + '/' + pos_list[i]
                next = i + 1
            elif label == 'S':
                yield char + '/' + pos_list[i]
                next = i + 1
        if next < len(text):
            yield text[next:] + '/' + pos_list[-1]

    # text 分词目标字符序列
    # state 状态枚举序列
    # start_p Pi_dic
    # trans_p A_dic
    # emit_p B_dic
    def viterbi(self, text, states, start_p, trans_p, emit_p):
        V = [{}]
        path = {}
        for y in states:
            V[0][y] = start_p[y] * emit_p[y].get(text[0], 0)
            path[y] = [y]
        for t in range(1, len(text)):
            V.append({})
            newpath = {}

            # 检验训练的发射概率矩阵中是否有该字
            not_exist = True
            for label, v in emit_p.items():
                not_exist = not_exist and text[t] not in v
            for y in states:
                emitP = emit_p[y].get(text[t], 0) if not not_exist else 1.0  # 设置未知字单独成词(类似于拉普拉斯平滑)
                (prob, state) = max([(V[t - 1][y0] * trans_p[y0].get(y, 0) * emitP, y0)
                                     for y0 in states if V[t - 1][y0] > 0])
                V[t][y] = prob
                newpath[y] = path[state] + [y]
            path = newpath
        if self.need_pos:
            if max([v.get(text[-1], 0) for k, v in emit_p.items() if 'M_' in k]) > max(
                    [v.get(text[-1], 0) for k, v in emit_p.items() if 'S_' in k]):
                (prob, state) = max(
                    [(V[len(text) - 1][y], y) for y, p in V[len(text) - 1].items() if 'M_' in y or 'S_' in y])
            else:
                (prob, state) = max([(V[len(text) - 1][y], y) for y in states])
        else:
            if emit_p['M'].get(text[-1], 0) > emit_p['S'].get(text[-1], 0):
                (prob, state) = max([(V[len(text) - 1][y], y) for y in ('E', 'M')])
            else:
                (prob, state) = max([(V[len(text) - 1][y], y) for y in states])

        return (prob, path[state])


if __name__ == '__main__':

    hmm = HMM(model_file='./model/hmm_model.pkl', dic_file='./dictionary/dictionary.txt', need_pos=False)
    res = list(hmm.cut("我想学习计算机编程"))
    print(res)

    cont = True
    while cont:
        text = str(input("请输入你想分词的句子： "))
        if text == "exit":
            cont = False
        else:
            res = hmm.cut(text)
            print(str(list(res)))
