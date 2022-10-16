# coding=UTF-8
# NNLM 神经网络语言模型，来自于A Neural Probabilistic Language Model
# 使用tensorflow实现
'''
File Name: nnlm_tf.py
Program IDE: PyCharm
Created Time: 2022/6/2 0002 22:36
Author: Wei Wei
'''

import tensorflow._api.v2.compat.v1 as tf
import numpy as np
from tensorflow.python.framework import ops
import glob
from word import *
import os

tf.logging.set_verbosity(tf.logging.ERROR)    # 设置tensorflow日志级别为info
tf.disable_v2_behavior()
ops.reset_default_graph()


class NNLM_TF():
    def __init__(self, corpus_file='./datasets/Chinese/*.txt', model_dir='./model/NNLM_TF_model', window=2,
                 sita=0.01):
        self.corpus_word_list = self.read_corpus_file(corpus_file)
        self.wf, self.id2word, self.word2id = self.word_process()
        self.model_dir = model_dir
        self.init_NNLM(window, sita)
        self.NNLM_train()

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
        self.n_step = window  # 窗口大小
        self.dim = 20  # 词向量维度
        self.n_hidden = 10  # 隐藏层大小
        self.V = len(self.word2id)  # 词库大小
        self.sita = sita
        self.input_batch, self.target_batch = self.make_batch()  # 初始化input输入词向量和target输出词hot-spot
        print("dictionary size: {0}".format(self.V))
        print("training input batch size: {0}".format(len(self.input_batch)))
        print("training target batch size: {0}".format(len(self.target_batch)))

    # 生成输入和标签
    def make_batch(self, words=None):
        if words == None:
            # 训练模式，用文本语料组装input和target
            input_batch = []
            target_batch = []
            for words in self.corpus_word_list:
                for i in range(self.n_step, len(words), 1):
                    input = [self.word2id[word] for word in words[i - self.n_step:i]]  # 将窗口之前的n个词加入input
                    target_id = self.word2id[words[i]]  # 将要预测的单词加入 target
                    input_batch.append(input)
                    # target hot-spot编码，在下面计算 softmax_cross_entropy_with_logits_v2 会用到
                    target = np.zeros((self.V))
                    target[target_id] = 1
                    target_batch.append(target)
        else:
            # 预测模式，用输入words做输入
            input_batch = []
            target_batch = None
            if len(words) < self.n_step:
                print("no enough input words")
            else:
                input = [self.word2id[word] for word in words[len(words) - self.n_step:len(words)]]
                input_batch.append(input)
        return input_batch, target_batch


    def NNLM_train(self):
        # 加载模型
        try:
            model_files = glob.glob(self.model_dir + '/NNLM_TF.*')
            if len(model_files) > 0:
                return
            else:
                raise FileNotFoundError
        except FileNotFoundError:
            print("NNLM model file doesn't exist. Need train.")
        # 重新训练模型
        X = tf.placeholder(tf.int64, [None, self.n_step])
        Y = tf.placeholder(tf.int64, [None, self.V])

        embedding = tf.get_variable(name='embedding', shape=[self.V, self.dim],
                                    initializer=tf.random_normal_initializer())  # 随机生成词向量
        XX = tf.nn.embedding_lookup(embedding, X)  # 词嵌入

        # 根据 y=b+X*W+tanh(d+X*H)*U 写代码
        input = tf.reshape(XX, shape=[-1, self.n_step * self.dim])
        H = tf.Variable(tf.random_normal([self.n_step * self.dim, self.n_hidden]), name='H')
        d = tf.Variable(tf.random_normal([self.n_hidden]), name='d')
        U = tf.Variable(tf.random_normal([self.n_hidden, self.V]), name='U')
        b = tf.Variable(tf.random_normal([self.V]), name='b')
        W = tf.Variable(tf.random_normal([self.n_step * self.dim, self.V]), name='W')
        A = tf.nn.tanh(tf.matmul(input, H) + d)
        B = tf.matmul(input, W) + tf.matmul(A, U) + b

        # 计算损失并进行优化
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=B, labels=Y))
        optimizer = tf.train.AdamOptimizer(self.sita).minimize(cost)
        # Adam即Adaptive Moment Estimation（自适应矩估计），是一个寻找全局最优点的优化算法，引入了二次梯度校正

        # tf 初始化
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        saver = tf.train.Saver()  # 初始化模型保存实例
        model_file = self.model_dir + '/NNLM_TF.ckpt'

        # 开始训练
        input_batch = self.input_batch
        target_batch = self.target_batch
        for epoch in range(500):
            _, loss = sess.run([optimizer, cost], feed_dict={X: input_batch, Y: target_batch})
            print('epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
            if (epoch + 1) % 100 == 0:
                saver.save(sess, model_file, global_step=epoch)
        # 保存模型
        saver.save(sess, model_file)
        print("Saved model checkpoint to {}\n".format(model_file))


    def predict_word(self, words):
        # tf 模型加载
        sess = tf.Session()
        # 加载训练好的模型
        # 图模型路径
        meta_path = self.model_dir + '/NNLM_TF.ckpt.meta'
        # 数据模型路径
        ckpt_path = self.model_dir + '/NNLM_TF.ckpt'
        # 导入图结构，加载.meta文件
        saver = tf.train.import_meta_graph(meta_path)
        # 恢复变量值，加载.ckpt文件
        saver.restore(sess, ckpt_path)
        graph = tf.get_default_graph()

        X = tf.placeholder(tf.int64, [None, self.n_step])
        Y = tf.placeholder(tf.int64, [None, self.V])

        embedding = graph.get_tensor_by_name('embedding:0')
        XX = tf.nn.embedding_lookup(embedding, X)  # 词嵌入

        # 根据 y=b+X*W+tanh(d+X*H)*U 写代码
        input = tf.reshape(XX, shape=[-1, self.n_step * self.dim])
        H = graph.get_tensor_by_name('H:0')
        d = graph.get_tensor_by_name('d:0')
        U = graph.get_tensor_by_name('U:0')
        b = graph.get_tensor_by_name('b:0')
        W = graph.get_tensor_by_name('W:0')
        A = tf.nn.tanh(tf.matmul(input, H) + d)
        B = tf.matmul(input, W) + tf.matmul(A, U) + b

        # 预测结果
        prediction = tf.argmax(B, 1)

        # 使用训练样本进行简单的预测
        input_batch, _ = self.make_batch(words)
        predict = sess.run([prediction], feed_dict={X: input_batch})
        return [self.id2word[i] for i in predict[0]]


if __name__ == '__main__':
    nnlm_tf = NNLM_TF()
    length = 20
    cont = True
    while cont:
        sentence = str(input("请输入你造句的主题句： "))
        if sentence == "exit":
            cont = False
        else:
            # 全模式分词，把句子中所有可以成词的词语都扫描出来，词语会重复，且不能解决歧义，适合关键词提取
            words = cut_rm_stopwords(sentence)
            if len(words) < nnlm_tf.n_step:
                print("最少2个词以上")
                continue
            for i in range(length):
                words.extend(nnlm_tf.predict_word(words))
            print(words)
