# coding=UTF-8
# NNLM 神经网络语言模型，来自于A Neural Probabilistic Language Model

import glob
import math
import pickle
import numpy as np
from word import *
import time
from util.simularity import *
from collections import Counter

# 激活函数, tanh为双曲正切函数，公式如下：
# tanh(x) = sinh(x) / cosh(x) = (exp(x) - exp(-x))/(exp(x) + exp(-x))
THRESHOLD = math.log(1e250)


class NNLM():
    def __init__(self, corpus_file='./datasets/Chinese/*.txt', model_file='./model/NNLM_model.pkl', window=4, sita=0.1):
        self.corpus_word_list = self.read_corpus_file(corpus_file)
        self.wf, self.id2word, self.word2id = self.word_process()
        self.model_file = model_file
        H, d, U, b = self.init_NNLM(window, sita)
        self.H, self.d, self.U, self.b, self.C = self.NNLM_train(H, d, U, b)
        self.word_vector = self.build_word_vec()

    def read_corpus_file(self, corpus_file):
        corpus_word_list = []
        files = glob.glob(corpus_file)
        total_file = len(files)
        i = 0
        load_corpus_progress = 0
        for filepath in files:
            with open(filepath, encoding='gbk', errors='ignore') as f:  # 读取需提取关键词的语料文件
                i += 1
                for line in f:
                    line = line.strip()  # 移除字符串头尾指定的字符（默认为空格或换行符）或字符序列
                    if not line: continue  # 跳过空行
                    word_pos_list = cut_rm_stopwords(line)
                    corpus_word_list.append(word_pos_list)  # 不限定词性为n开头的词
            f.close()
            if load_corpus_progress < round(i * 100 / total_file):
                load_corpus_progress = round(i * 100 / total_file)
                print("corpus file load finish rate is: {0}%".format(load_corpus_progress))
        return corpus_word_list

    def word_process(self):
        wf = dict()  # corpus中，每个词的词频
        id2word = dict()
        word2id = dict()
        for corpus_word in self.corpus_word_list:
            for word in corpus_word:
                if word not in word2id.keys():
                    word2id[word] = len(word2id)
                    id2word[word2id[word]] = word
                wf[word] = wf.get(word, 0) + 1
        return wf, id2word, word2id

    # 初始化神经网络语言模型：y = b + U * tanh（d + Hx）
    # 输入层：input_cells：输入层神经元个数，n：词特征选取窗口，(n-1)*input_cells：输入层
    # 隐层：hidden_cells：隐层神经元个数，也是词向量，维度为hidden_cells
    # 输出层：output_cells = len(word_dictionary)：输出层神经元个数，也是词典向量，词典向量维度为词典大小。
    # 输入层到隐层的转换矩阵：H：h*((n-1)*m)
    # 隐层到输出层的转换矩阵：U: |V| * h
    # 特征矩阵C：|V|*m
    # 入参iter为迭代上限次数
    def init_NNLM(self, window, sita):
        self.input_cells = 10
        self.hidden_cells = 10
        self.output_cells = len(self.word2id)
        self.n = window
        self.sita = sita
        H = np.random.rand(self.hidden_cells, (self.n - 1) * self.input_cells)
        d = np.random.rand(self.hidden_cells)
        U = np.random.rand(self.output_cells, self.hidden_cells)
        b = np.random.rand(self.output_cells)
        # W = np.random.rand(self.output_cells, (self.n - 1) * self.input_cells)
        return H, d, U, b

    def NNLM_train(self, H, d, U, b, iter=50):
        # 加载模型
        try:
            H, d, U, b, C = self.load_model()
            if C is None or len(C.keys()) == 0:
                raise FileNotFoundError
            return H, d, U, b, C
        except FileNotFoundError:
            print("NNLM model file doesn't exist. Need train.")
        # 初始化空词向量
        blank_vector = np.zeros((self.input_cells))
        # 词典映射的特征矩阵C，|V|*input_cells维度，
        C = dict()
        # 初始化词典中的词向量
        for word in self.wf.keys():
            C[word] = np.random.rand(self.input_cells)
        # 开始NNLM的训练
        start_time = time.time()
        print('[%s] Start training' % datetime.datetime.now())
        r = 0
        while r < iter:
            r += 1
            print("Round: {0}".format(r))
            L = []
            l = 0
            for line in self.corpus_word_list:
                l += 1
                print("current line: {0}".format(l))
                for w in range(len(line)):
                    # 根据窗口window选取前后关联词，构建输入X
                    X = []
                    inputwords = []
                    for i in range(w - 1, w - self.n, -1):
                        if i < 0:
                            X.extend(blank_vector)
                        else:
                            X.extend(C[line[i]])
                            inputwords.append(line[i])
                    # 前向计算
                    X = np.array(X)
                    # 计算公式：y = b + U * tanh（d + Hx）
                    # 计算隐层的输入向量o
                    o = np.dot(X, H.T) + d
                    # 计算隐层的输出向量a
                    a = np.tanh(o)
                    # 计算输出层输入向量y
                    # y = np.dot(U, a) + np.dot(W, X) + b
                    y = np.dot(a, U.T) + b
                    # 计算NNLM的输出，softmax函数做归一化
                    # if np.max(y) >= THRESHOLD:
                    #     y = y - THRESHOLD  # 减去一个阈值，避免exp运算溢出，分母做了归一化之后，不影响实际概率计算结果
                    p = np.exp(y)
                    s = sum(p)
                    p = p / s
                    # 计算L = logpwt
                    if p[self.word2id[line[w]]] != 0:
                        L.append(math.log(p[self.word2id[line[w]]]))
                    else:
                        L.append(1e-250)
                    # print("似然函数L = {0}".format(L * -1))
                    # 反向修正参数
                    # 计算及更新输出层参数
                    La = np.zeros((self.hidden_cells))
                    Ly = np.zeros((self.output_cells))
                    Lx = np.zeros(((self.n - 1) * self.input_cells))
                    for j in range(self.output_cells):
                        if j == self.word2id[line[w]]:
                            Ly[j] = 1 - p[j]
                        else:
                            Ly[j] = -1 * p[j]
                        # Lx = Lx + (Ly[j] * W[j, :]).reshape(-1, 1)
                        La = La + Ly[j] * U[j, :]
                        # W[j, :] = W[j, :] + np.ravel(self.sita * Ly[j] * X.T)
                        U[j, :] = U[j, :] + self.sita * Ly[j] * a
                    b = b + self.sita * Ly
                    # 计算及更新隐层
                    Lo = np.zeros((self.hidden_cells))
                    Lo = (1 - a * a) * La
                    Lx = Lx + np.dot(Lo, H)
                    d = d + self.sita * Lo
                    H = H + self.sita * np.dot(Lo.reshape(-1, 1), X.reshape(1, -1))
                    # 计算及更新输入层，词典特征矩阵
                    k = 0
                    X = np.array(X + self.sita * Lx).reshape(self.n - 1, self.input_cells)
                    for i in range(w - 1, w - self.n, -1):
                        if i < 0:
                            k += 1
                        else:
                            C[line[i]] += self.sita * X[k, :]
                            k += 1
            L = np.array(L)
            print("L_max: {0}, L_min: {1}, L_avg: {2}".format(np.max(L), np.min(L), np.average(L)))
            iter_lapsed_time = time.time() - start_time
            print('round lapsed time: %f' % iter_lapsed_time)
        # 保存模型参数
        model_file = open(self.model_file, 'wb')
        pickle.dump(H, model_file)
        pickle.dump(d, model_file)
        pickle.dump(U, model_file)
        pickle.dump(b, model_file)
        # pickle.dump(W, model_file)
        pickle.dump(C, model_file)
        model_file.close()
        elapsed_time = time.time() - start_time
        print('* Elapsed time: %f' % elapsed_time)
        print('* [%s] Training done' % datetime.datetime.now())
        return H, d, U, b, C

    def load_model(self):
        model_file = open(self.model_file, 'rb')
        H = pickle.load(model_file)
        d = pickle.load(model_file)
        U = pickle.load(model_file)
        b = pickle.load(model_file)
        # W = pickle.load(model_file)
        C = pickle.load(model_file)
        model_file.close()
        return H, d, U, b, C

    def build_word_vec(self):
        # 把C转成矩阵格式
        word_vec = np.zeros((len(self.C), self.input_cells))
        for word, vector in self.C.items():
            word_vec[self.word2id[word], :] = vector
        return word_vec

    def predict_word(self, sentence):
        l = len(sentence)
        word_list = []
        if l >= self.n:
            word_list = sentence[l - (self.n - 1):]
        else:
            word_list = sentence
        l = len(word_list)
        # 初始化空词向量
        blank_vector = np.zeros((self.input_cells))
        X = []
        inputwords = []
        for i in range(l - 1, l - self.n, -1):
            if i < 0:
                X.extend(blank_vector)
            else:
                X.extend(self.C[word_list[i]])
                inputwords.append(word_list[i])
        # 前向计算
        X = np.array(X)
        # 计算公式：y = b + U * tanh（d + Hx）
        # 计算隐层的输入向量o
        o = np.dot(X, self.H.T) + self.d
        # 计算隐层的输出向量a
        a = np.tanh(o)
        # 计算输出层输入向量y
        # y = np.dot(U, a) + np.dot(W, X) + b
        y = np.dot(a, self.U.T) + self.b
        # 计算NNLM的输出，softmax函数做归一化
        p = np.exp(y)
        s = sum(p)
        p = p / s
        exist = True
        while exist:
            outputword = self.id2word[np.where(p == np.max(p))[0][0]]
            if outputword in sentence:
                p = np.delete(p, np.where(p == np.max(p))[0][0])
            else:
                exist = False
        return outputword


if __name__ == '__main__':
    nnlm = NNLM()
    C = nnlm.C
    id2word = nnlm.id2word
    word2id = nnlm.word2id
    word_vec = nnlm.word_vector
    id_list = [id for word, id in word2id.items()]
    id_list = np.array(id_list)
    test_word = '希望'
    test_word_vec = C[test_word]
    print("'{0}'的词向量为：{1}".format(test_word, test_word_vec))
    # 通过计算向量距离，找到同义词
    dist, ids = getKNN(C[test_word], word_vec, id_list, 2)
    synonym = [id2word[id] for id in ids]
    print("通过向量距离计算，得到'{0}'的同义词为：{1}".format(test_word, synonym))
    # 通过计算余弦相似度，找到余弦最接近1的词典中的词，作为同义词
    max_cos_sim = -1 * float('inf')
    synonym = ''
    for i in range(word_vec.shape[0]):
        if i == word2id[test_word]:
            continue
        else:
            cos_sim = calc_cos_sim(test_word_vec, np.squeeze(word_vec[i, :]))
            if max_cos_sim < cos_sim:
                max_cos_sim = cos_sim
                synonym = id2word[i]
    print("通过余弦相似度计算，得到'{0}'的同义词为：{1}".format(test_word, synonym))

    # NLG
    sentence = ['教师']
    lenth = 10
    for w in range(lenth):
        next_word = nnlm.predict_word(sentence)
        sentence.append(next_word)
    print(sentence)
    counter = Counter(nnlm.wf)
