# coding=UTF-8
# 基于TF-IDF，LSA，LDA主题算法，进行关键词提取任务

import math

from gensim import corpora, models
from jieba import analyse
import functools
from word import *
from math import log
import numpy as np
from util import simularity as sm


#  排序函数，用于topK关键词的按值排序
def cmp(e1, e2):
    res = np.sign(e1[1] - e2[1])
    if res != 0:
        return res
    else:
        a = e1[0] + e2[0]
        b = e2[0] + e1[0]
        if a > b:
            return 1
        elif a == b:
            return 0
        else:
            return -1


# TF-IDF类
class Tf_Idf(object):

    # 四个参数分别是：训练好的idf字典，默认idf值，处理后的待提取文本，关键词数量
    def __init__(self, corpus_file='./datasets/corpus.txt', keyword_num=10):
        self.corpus_file = corpus_file
        self.doc_word_list = list()
        self.keyword_num = keyword_num

    def read_corpus_file(self):
        self.D = 0  # 文档集中的总文档数
        with open(self.corpus_file, encoding='utf-8') as f:  # 读取需提取关键词的语料文件
            for line in f:
                line = line.strip()  # 移除字符串头尾指定的字符（默认为空格或换行符）或字符序列
                if not line: continue  # 跳过空行
                word_pos_list = cut_with_pos_rm_sw_list(line)
                self.doc_word_list.append(word_pos_filter(word_pos_list))
                self.D += 1
        f.close()

    # 计算词word_ij的tf，定义：词i在文档j中出现的频次
    # 公式为: tf_ij = n_ij / sum_i(n_ij)
    def calc_tf(self):
        # tf_dic的格式为：{j, {词i：tf_value}}
        self.tf = dict()
        j = 1
        for words in self.doc_word_list:
            self.tf[j] = dict()
            n = 0
            for word in words:
                self.tf[j][word] = self.tf[j].get(word, 0.0) + 1.0
                n += 1
            for word, count in self.tf[j].items():
                self.tf[j][word] = count / n

    @staticmethod
    def calc_tf(word_list):
        tf_dic = dict()
        for word in word_list:
            tf_dic[word] = tf_dic.get(word, 0.0) + 1.0
        for word, count in tf_dic.items():
            tf_dic[word] = count / len(word_list)
        return tf_dic

    # 计算词word_ij的idf, 定义：文档|D_i|为文档集中出现词i的文档数量
    # 公式为：idf_i = log(|D| / (1 + |Di|))
    def calc_idf(self):
        self.idf = dict()
        all_words = []  # 保留所有文档集中所有的词，不重复
        for words in self.doc_word_list:
            for word in words:
                if word not in all_words:
                    all_words.append(word)
        for word in all_words:
            Di = 0
            for words in self.doc_word_list:
                if word in words:
                    Di += 1
            self.idf[word] = log(self.D / (1 + Di))
        self.default_idf = log(self.D / 1)  # 主要用于训练集中没有出现的词，做拉普拉斯平滑

    # 计算词word_ij的tf_idf.
    # 公式为：tf_idf_ij = nij / sum_i(nij) * log(|D| / (1 + |Di|))
    def calc_tf_idf(self):
        for j in self.tf.keys():
            for word, tf_value in self.tf[j]:
                self.tf[j][word] = tf_value * self.idf[word]

    # 基于corpus训练数据集，计算idf值
    def model_train(self):
        self.read_corpus_file()
        self.calc_idf()

    # 根据测试文本text，计算tf-idf, 并提取关键词
    def get_keywords(self, text, keyword_num):
        self.keyword_num = keyword_num
        word_pos_list = cut_with_pos_rm_sw_list(text)
        word_list = word_pos_filter(word_pos_list)
        tf_dic = self.calc_tf(word_list)
        tf_idf_dic = dict()
        for word, tf_value in tf_dic.items():
            idf_value = self.idf.get(word, self.default_idf)
            tf_idf_dic[word] = tf_value * idf_value
        keywords = []
        for word, _ in sorted(tf_idf_dic.items(), key=functools.cmp_to_key(cmp), reverse=True)[:self.keyword_num]:
            keywords.append(word)
        return keywords


# TextRank模型
class TextRank(object):
    def __init__(self, windows=5, iter=100):
        self.windows = windows  # 切分窗口大小
        self.iter = iter  # 计算每个词得分的迭代次数
        self.d = 0.99  # 阻尼系数，用于孤立词打分计算。

    def keyword_byTextRank(self, text, keyword_num=10):
        word_list = cut_rm_stopwords(text)
        # 构建每个词的入链词列表。格式：{词：{score:xx, In_word:{入链词set}， Out_word:{入链词set}}}
        word_link = dict()
        # step1：初始化{词：score}词典
        for word in word_list:
            if word not in word_link.keys():
                word_link[word] = dict()
                word_link[word]['score'] = 1
                word_link[word]['In_words'] = []
                word_link[word]['Out_words'] = []
        # step2：构建入链词和出链词
        i = 0
        for word in word_list:
            if i + (self.windows - 1) < len(word_list):
                window_word = [w for w in word_list[i:i + self.windows] if w != word]
            elif i + (self.windows - 1) >= len(word_list) and len(word_list) - self.windows >= 0:
                window_word = [w for w in word_list[-5:-1] if w != word]
            else:
                window_word = [w for w in word_list if w != word]
            for w in window_word:
                word_link[word]['In_words'].append(w)
                word_link[word]['Out_words'].append(w)
                word_link[w]['Out_words'].append(word)
                word_link[w]['In_words'].append(word)
            i += 1

        # 迭代计算每个词的得分，基于每个入链词的得分平均贡献给当前词，最大迭代次数为iter
        n = 0
        while n < self.iter:
            for word, textrank in word_link.items():
                score = textrank['score']
                In_words = textrank['In_words']
                for in_word in In_words:
                    in_word_score = word_link[in_word]['score']
                    if in_word_score > 0:
                        score += in_word_score / len(word_link[in_word]['Out_words'])
                        in_word_score -= in_word_score / len(word_link[in_word]['Out_words'])
                        word_link[in_word]['score'] = in_word_score
                word_link[word]['score'] = (1 - self.d) + self.d * score
            n += 1

        # 按照score值排序，输出前keyword_num个数的关键词
        word_score = dict()
        for word, textrank in word_link.items():
            word_score[word] = word_score.get(word, textrank['score'])
        keywords = []
        for word, _ in sorted(word_score.items(), key=functools.cmp_to_key(cmp), reverse=True)[:keyword_num]:
            keywords.append(word)
        return keywords

    @staticmethod
    def keyword_byTextRank_2(text, pos=False, keyword_num=10):
        textrank = analyse.textrank
        keywords = textrank(text, keyword_num)
        return keywords


# 主题模型
class TopicModel(object):
    # 三个传入参数：处理后的数据集，关键词数量，具体模型（LSI、LDA），主题数量
    def __init__(self, corpus_file, keyword_num=10, modelType='LSI', topic_num=4):
        self.doc_word_list, self.D = self.read_corpus_file(corpus_file)
        self.modelType = modelType  # 主题模型的两种算法：LSI——潜在语义分析，LDA——隐含狄利克雷分析
        self.keyword_num = keyword_num  # 最终输出的关键词数量
        self.topic_num = topic_num  # 隐含主题数量
        # 主题模型训练
        self.topic_model_train()

    @staticmethod
    def read_corpus_file(corpus_file):
        D = 0  # 文档集中的总文档数
        doc_word_list = []
        with open(corpus_file, encoding='utf-8') as f:  # 读取需提取关键词的语料文件
            for line in f:
                line = line.strip()  # 移除字符串头尾指定的字符（默认为空格或换行符）或字符序列
                if not line: continue  # 跳过空行
                word_pos_list = cut_rm_stopwords(line)
                # doc_word_list.append(word_pos_filter(word_pos_list))  # 限定词性为n开头的词
                doc_word_list.append(word_pos_list)  # 不限定词性为n开头的词
                D += 1
        f.close()
        return doc_word_list, D

    def topic_model_train(self):
        # 使用gensim的接口，将文本转为向量化表示
        # 先构建词空间
        self.dictionary = corpora.Dictionary(self.doc_word_list)
        # 使用BOW模型向量化，BOW: Bag of Word, BOW的格式：[[(word_id, num)]]
        corpus = [self.dictionary.doc2bow(doc) for doc in self.doc_word_list]
        # 对每个词，根据tf-idf进行加权，得到加权后的向量表示
        self.tfidf_model = models.TfidfModel(corpus)
        self.corpus_tfidf = self.tfidf_model[corpus]
        # 选择加载的模型
        if self.modelType == 'LSI':
            self.model = self.train_lsi()
        else:
            self.model = self.train_lda()

        # 得到数据集的主题-词分布
        word_dic = self.word_dictionary(self.doc_word_list)
        self.wordtopic_dic = self.get_wordtopic(word_dic)

    def train_lsi(self):
        """
        def __init__(self, corpus=None, num_topics=200, id2word=None, chunksize=20000, decay=1.0, distributed=False,
                     onepass=True, power_iters=P2_EXTRA_ITERS, extra_samples=P2_EXTRA_DIMS)
        corpus: 语料，如果指定了语料就使用语料，或者可用add_document这个方法去增加语料。
        num_topics: 分解后的主题数
        onepass: 表示随机算法采用是否是一路或多路；onepass = True表示为单路算法，否则为多路算法。
        id2word: 词id到词的映射；
        chunksize: 在训练过程中，一次训练文档数。chunksize的大小是速度与内存的一个折衷。如果是分布式运行环境，每个chunk会被发送到不同的工作节点。
        decay：当文档更新时，希望模型更顷向于哪个语料训练的结果，是旧的语料还是现在所给的新的语料 ；
        distributed：是否是分布式计算；
        `power_iters` 与 `extra_samples` 是算法里面的参数，会影响随机多路算法的正确性，当onepass = True内部使用，否则使用前端算法；
        `power_iters` and `extra_samples` affect the accuracy of the stochastic multi - pass algorithm, which is used either internally(`onepass = True `) or as the front - end algorithm(`onepass = False `).
        """
        lsi = models.LsiModel(self.corpus_tfidf, id2word=self.dictionary, num_topics=self.topic_num)
        # 构造word_doc matrix
        word_doc_mat = np.zeros((len(self.dictionary.token2id), len(self.doc_word_list)))
        for i in range(len(self.doc_word_list)):
            for word in self.doc_word_list[i]:
                word_doc_mat[self.dictionary.token2id[word]][i] += 1
        self.word_doc_mat = word_doc_mat.astype(float)
        # 对word_doc matrix进行SVD（奇异值分解）
        self.u, self.sigma, self.vT = np.linalg.svd(self.word_doc_mat, full_matrices=False)
        # 根据选取的主题数，计算word——topic映射矩阵，以及topic——doc映射矩阵
        topic_mat = np.diag(self.sigma[:self.topic_num])
        self.word_topic_mat = np.dot(self.u[:, :self.topic_num], topic_mat)
        self.topic_doc_mat = np.dot(topic_mat, self.vT[:self.topic_num, :])
        return lsi

    def train_lda(self):
        lda = models.LdaModel(self.corpus_tfidf, id2word=self.dictionary, num_topics=self.topic_num)
        return lda

    def get_wordtopic(self, word_dic):
        wordtopic_dic = {}
        for word in word_dic:
            single_list = [word]
            wordcorpus = self.tfidf_model[self.dictionary.doc2bow(single_list)]
            wordtopic = self.model[wordcorpus]
            wordtopic_dic[word] = wordtopic
        return wordtopic_dic

    # 计算词的分布和文档的分布的相似度，取相似度最高的keyword_num个词作为关键词
    def extract_keyword(self, text):
        word_list = cut_rm_stopwords(text)
        word_doc_mat = np.zeros((len(self.dictionary.token2id),))
        for word in word_list:
            if word in self.dictionary.token2id.keys():
                word_doc_mat[self.dictionary.token2id[word]] += 1
        # 计算输入文档与训练集文档相似度，选取相似度最高（最接近1）的文档
        max_cos = 0
        max_i = 0
        for i in range(self.word_doc_mat.shape[1]):
            cos = sm.calc_cos_sim(word_doc_mat, np.ravel(self.word_doc_mat[:, i]))
            if max_cos < cos:
                max_cos = cos
                max_i = i
        doc_topic = np.ravel(self.topic_doc_mat[:, max_i])
        # 计算每个词与此topic的相似度，取出相似度最高（最接近1）的词
        word_topic_mat = word_doc_mat.reshape(-1, 1) * self.word_topic_mat
        word_topic_sim = {}
        for i in range(word_topic_mat.shape[0]):
            word = self.dictionary.id2token[i]
            word_topic_sim[word] = sm.calc_cos_sim(np.ravel(word_topic_mat[i, :]), doc_topic)
        # 排序，取与文档主题最契合的，前num个词做为关键词
        keywords = []
        for k, v in sorted(word_topic_sim.items(), key=functools.cmp_to_key(cmp), reverse=True)[:self.keyword_num]:
            keywords.append(k)
        return keywords

    # 计算词的分布和文档的分布的相似度，取相似度最高的keyword_num个词作为关键词
    def extract_keyword_2(self, text):
        word_list = cut_rm_stopwords(text)
        sentcorpus = self.tfidf_model[self.dictionary.doc2bow(word_list)]
        senttopic = self.model[sentcorpus]

        # 计算输入文本和每个词的主题分布相似度
        sim_dic = {}
        for k, v in self.wordtopic_dic.items():
            if k not in word_list:
                continue
            sim = sm.calsim(v, senttopic)
            sim_dic[k] = sim
        keywords = []
        for k, v in sorted(sim_dic.items(), key=functools.cmp_to_key(cmp), reverse=True)[:self.keyword_num]:
            keywords.append(k)
        return keywords

    # 词空间构建方法和向量化方法，在没有gensim接口时的一般处理方法
    def word_dictionary(self, doc_list):
        dictionary = []
        for doc in doc_list:
            dictionary.extend(doc)

        dictionary = list(set(dictionary))

        return dictionary

    def doc2bowvec(self, word_list):
        vec_list = [1 if word in word_list else 0 for word in self.dictionary]
        return vec_list


if __name__ == '__main__':
    text = '6月19日,《2012年度“中国爱心城市”公益活动新闻发布会》在京举行。' + \
           '中华社会救助基金会理事长许嘉璐到会讲话。基金会高级顾问朱发忠,全国老龄' + \
           '办副主任朱勇,民政部社会救助司助理巡视员周萍,中华社会救助基金会副理事长耿志远,' + \
           '重庆市民政局巡视员谭明政。晋江市人大常委会主任陈健倩,以及10余个省、市、自治区民政局' + \
           '领导及四十多家媒体参加了发布会。中华社会救助基金会秘书长时正新介绍本年度“中国爱心城' + \
           '市”公益活动将以“爱心城市宣传、孤老关爱救助项目及第二届中国爱心城市大会”为主要内容,重庆市' + \
           '、呼和浩特市、长沙市、太原市、蚌埠市、南昌市、汕头市、沧州市、晋江市及遵化市将会积极参加' + \
           '这一公益活动。中国雅虎副总编张银生和凤凰网城市频道总监赵耀分别以各自媒体优势介绍了活动' + \
           '的宣传方案。会上,中华社会救助基金会与“第二届中国爱心城市大会”承办方晋江市签约,许嘉璐理' + \
           '事长接受晋江市参与“百万孤老关爱行动”向国家重点扶贫地区捐赠的价值400万元的款物。晋江市人大' + \
           '常委会主任陈健倩介绍了大会的筹备情况。'


    tf_idf = Tf_Idf()
    tf_idf.model_train()
    keywords = tf_idf.get_keywords(text, 10)
    print('TF-IDF模型抽取关键字如下：')
    print(keywords)

    print('TextRank模型结果：')
    textrank = TextRank()
    keywords = textrank.keyword_byTextRank(text)
    print("自建模型的输出结果为：{0}".format(keywords))
    keywords = textrank.keyword_byTextRank_2(text)
    print("jieba.analyse模型的输出结果为：{0}".format(keywords))

    print('主题LSI模型结果：')
    topicModel = TopicModel('datasets/corpus.txt', 10, 'LSI', 10)
    print("主题LSI模型，提取关键词的结果：")
    keywords = topicModel.extract_keyword(text)
    print(keywords)
    print("Gensim自带的LSI模型，提取关键词的结果：")
    keywords = topicModel.extract_keyword_2(text)
    print(keywords)
    print('主题LDA模型结果：')
    topicModel = TopicModel('datasets/corpus.txt', 10, 'LDA')
    keywords = topicModel.extract_keyword_2(text)
    print(keywords)
