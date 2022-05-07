# coding=UTF-8
# 隐马尔科夫概率模型分词

class HMM(object):
    def __init__(self):
        import os

        # 主要是用于存储算法模型的结果，是训练模式还是测试模式
        self.dic_file = './dictionary/dictionary.txt'

        # 主要是用于存储算法模型的结果，是训练模式还是测试模式
        self.model_file = './model/hmm_model.pkl'

        # 模型状态枚举值
        self.state_list = ['B', 'M', 'E', 'S']

        # 参数加载，用于判断是否需要加载模型文件
        self.load_model = False

    # 用于加载已计算的模型结果，当需要重新训练时，需初始化
    def try_load_model(self, trained):
        if trained:
            import pickle
            with open(self.model_file, 'rb') as f:
                self.A_dic = pickle.load(f)  # 状态转移概率
                self.B_dic = pickle.load(f)  # 发射概率概率
                self.Pi_dic = pickle.load(f)  # 状态的初始概率
                self.load_model = True
        else:
            # 状态转移概率（状态->状态的条件概率），字典类型，格式如下：
            #     B   M   E   S
            # B   0.0 0.0 0.0 0.0
            # M   0.0 0.0 0.0 0.0
            # E   0.0 0.0 0.0 0.0
            # S   0.0 0.0 0.0 0.0
            self.A_dic = {}
            # 发射概率（状态->字的条件概率），字典类型，格式为：{'B':{}, 'M':{}, 'E':{}, 'S':{}}
            #     字1  字2 字3 字4
            # B   0.0 0.0 0.0 0.0
            # M   0.0 0.0 0.0 0.0
            # E   0.0 0.0 0.0 0.0
            # S   0.0 0.0 0.0 0.0
            self.B_dic = {}
            # 状态的初始概率（状态到首字的概率），字典类型，格式为：{'B':num, 'M':num, 'E':num, 'S':num}
            self.Pi_dic = {}
            self.load_model = False

    # 通过给定的分词语料进行训练，计算转移概率、发射概率以及初始概率
    def train(self, path='./dictionary/dictionary.txt'):
        # 重置模型概率
        self.try_load_model(False)

        # 统计状态出现次数，求p(o)，字典类型
        Count_dic = {}

        # 初始化训练参数
        def init_parameters():
            for state in self.state_list:
                #     B   M   E   S
                # B   0.0 0.0 0.0 0.0
                # M   0.0 0.0 0.0 0.0
                # E   0.0 0.0 0.0 0.0
                # S   0.0 0.0 0.0 0.0
                self.A_dic[state] = {s: 0.0 for s in self.state_list}
                # {'B':0.0, 'M':0.0, 'E':0.0, 'S':0.0}
                self.Pi_dic[state] = 0.0
                # {'B':{}, 'M':{}, 'E':{}, 'S':{}}
                self.B_dic[state] = {}

                Count_dic[state] = 0

        def makeLabel(text):
            out_text = []
            if len(text) == 1:
                out_text.append('S')
            else:
                out_text += ['B'] + ['M'] * (len(text) - 2) + ['E']

            return out_text

        init_parameters()
        line_num = 0

        # 观察者集合，主要是字和标点等
        words = set()
        with open(path, encoding='utf8') as f:  # 读取字典文件
            for line in f:
                line_num += 1
                # if line_num > 1000:
                #     break
                line = line.strip()  # 移除字符串头尾指定的字符（默认为空格或换行符）或字符序列
                # print(line)
                if not line:
                    continue
                word_list = [i for i in line if i != ' ']  # 字列表，过滤中间的空字
                words |= set(word_list)  # 更新字的集合，set取并集
                # print("word_list: " + str(word_list))
                linelist = line.split()
                # print("linelist: " + str(linelist))
                line_state = []
                for w in linelist:
                    line_state.extend(makeLabel(w))

                assert len(word_list) == len(line_state)
                # print("line_state: " + str(line_state))
                for k, v in enumerate(line_state):
                    Count_dic[v] += 1  # state序列中，某个state出现的次数，从整个字典文件角度统计
                    if k == 0:
                        self.Pi_dic[v] += 1  # 每个句子的第一个字的状态，用于计算初始状态概率
                    else:
                        # 状态->状态的转移次数，用于计算转移概率
                        self.A_dic[line_state[k - 1]][v] += 1
                        # 状态->词语的转移次数，用于计算发射概率，默认值为0
                        self.B_dic[line_state[k]][word_list[k]] = \
                            self.B_dic[line_state[k]].get(word_list[k], 0) + 1.0

        self.Pi_dic = {k: v * 1.0 / line_num for k, v in self.Pi_dic.items()}
        self.A_dic = {k: {k1: v1 / Count_dic[k] for k1, v1 in v.items()}
                      for k, v in self.A_dic.items()}
        # 加1平滑
        self.B_dic = {k: {k1: (v1 + 1) / Count_dic[k] for k1, v1 in v.items()}
                      for k, v in self.B_dic.items()}

        import pickle
        with open(self.model_file, 'wb') as file:
            pickle.dump(self.A_dic, file)
            pickle.dump(self.B_dic, file)
            pickle.dump(self.Pi_dic, file)

        return self

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
            neverSeen = text[t] not in emit_p['S'].keys() and \
                        text[t] not in emit_p['M'].keys() and \
                        text[t] not in emit_p['E'].keys() and \
                        text[t] not in emit_p['B'].keys()
            for y in states:
                # 设置未知字单独成词
                # if neverSeen:
                #     emitP = 1.0
                # else:
                #     emitP = emit_p[y].get(text[t], 0)
                emitP = emit_p[y].get(text[t], 0) if not neverSeen else 1.0  # 设置未知字单独成词
                (prob, state) = max([(V[t - 1][y0] * trans_p[y0].get(y, 0) * emitP, y0)
                                     for y0 in states if V[t - 1][y0] > 0])
                V[t][y] = prob
                newpath[y] = path[state] + [y]
            path = newpath

        if emit_p['M'].get(text[-1], 0) > emit_p['S'].get(text[-1], 0):
            (prob, state) = max([(V[len(text) - 1][y], y) for y in ('E', 'M')])
        else:
            (prob, state) = max([(V[len(text) - 1][y], y) for y in states])

        return (prob, path[state])

    def cut(self, text):
        import os
        if not self.load_model:
            self.try_load_model(os.path.exists(self.model_file))
        prob, pos_list = self.viterbi(text, self.state_list, self.Pi_dic, self.A_dic, self.B_dic)
        begin, next = 0, 0
        for i, char in enumerate(text):
            pos = pos_list[i]
            if pos == 'B':
                begin = i
            elif pos == 'E':
                yield text[begin: i + 1]
                next = i + 1
            elif pos == 'S':
                yield char
                next = i + 1
        if next < len(text):
            yield text[next:]


if __name__ == '__main__':

    hmm = HMM()
    hmm.train()

    cont = True
    while cont:
        text = str(input("请输入你想分词的句子： "))
        if text == "exit":
            cont = False
        else:
            res = hmm.cut(text)
            print(str(list(res)))
