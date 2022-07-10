# coding=UTF-8
# 非负矩阵分解算法（Non-negative Matrix Factorizations）
# pLSA(probabilistic Latent Semantic Analysis)，概率潜在语义分析模型，
# 是1999年Hoffman提出的一个被称为第一个能解决一词多义问题的模型，
# 通过在文档与单词之间建立一层主题（Topic），将文档与单词的直接关联转化为文档与主题的关联以及主题与单词的关联。
# 这里采用EM算法进行估计

import math
import random
import jieba
import codecs
import datetime


class pLSA():
    def __init__(self, ntopic=5):
        self.n_doc = 0
        self.n_word = 0
        self.n_topic = ntopic
        self.corpus = None
        self.p_z_dw = None
        self.p_w_z = None
        self.p_z_d = None
        self.likelihood = 0
        self.vocab = None
        self.stop_words = [u'，', u'。', u'、', u'（', u'）', u'·', u'！', u' ', u'：', u'“', u'”', u'\n']

    # 每行和为1的正实数，概率分布;
    def _rand_mat(self, sizex, sizey):
        ret = []
        for i in range(sizex):
            ret.append([])
            for _ in range(sizey):
                ret[-1].append(random.random())
            norm = sum(ret[-1])
            for j in range(sizey):
                ret[-1][j] /= norm
        return ret

    # 从文本中计算词频稀疏矩阵，这里存储模型仿照LDA
    def loadCorpus(self, fn):
        # 中文分词
        f = open(fn, 'r', encoding='UTF-8')
        text = f.readlines()
        text = r' '.join(text)
        seg_generator = jieba.cut(text)
        seg_list = [i for i in seg_generator if i not in self.stop_words]
        seg_list = r' '.join(seg_list)
        # 切割统计所有出现的词纳入词典
        seglist = seg_list.split(" ")
        self.vocab = []
        for word in seglist:
            if (word != u' ' and word not in self.vocab):
                self.vocab.append(word)
        self.n_word = len(self.vocab)
        CountMatrix = []
        f.seek(0, 0)
        # 统计每个文档中出现的词频
        for line in f:
            # 置零
            count = [0 for i in range(len(self.vocab))]
            text = line.strip()
            # 但还是要先分词
            seg_generator = jieba.cut(text)
            seg_list = [i for i in seg_generator if i not in self.stop_words]
            seg_list = r' '.join(seg_list)
            seglist = seg_list.split(" ")
            # 查询词典中的词出现的词频
            for word in seglist:
                if word in self.vocab:
                    count[self.vocab.index(word)] += 1
            CountMatrix.append(count)
        f.close()
        self.corpus = CountMatrix
        self.n_doc = len(CountMatrix)
        # 初始化
        self.p_z_d = self._rand_mat(self.n_topic, self.n_doc)
        self.p_w_z = self._rand_mat(self.n_word, self.n_topic)
        self.p_z_dw = []
        for k in range(self.n_topic):
            self.p_z_dw.append(self._rand_mat(self.n_doc, self.n_word))

    def _e_step(self):
        for k in range(self.n_topic):
            for d in range(self.n_doc):
                for j in range(self.n_word):
                    _d_wz_zd = 0
                    for kk in range(self.n_topic):
                        _d_wz_zd += self.p_w_z[j][kk] * self.p_z_d[kk][d]
                    if _d_wz_zd <= 0:
                        _d_wz_zd = 1e-6
                    self.p_z_dw[k][d][j] = self.p_w_z[j][k] * self.p_z_d[k][d] / _d_wz_zd

    def _m_step(self):
        print("updating Pn(Wj|Zk)...\r")
        for j in range(self.n_word):
            for k in range(self.n_topic):
                _d_dw_zdw = 0
                for d in range(self.n_doc):
                    _d_dw_zdw += self.corpus[d][j] * self.p_z_dw[k][d][j]
                _d_dw_zdw_sum = 0
                for jj in range(self.n_word):
                    _d_dw_zdw_i = 0
                    for d in range(self.n_doc):
                        _d_dw_zdw_i += self.corpus[d][jj] * self.p_z_dw[k][d][jj]
                    _d_dw_zdw_sum += _d_dw_zdw_i
                if _d_dw_zdw_sum <= 0:
                    _d_dw_zdw_sum = 1e-6
                self.p_w_z[j][k] = _d_dw_zdw / _d_dw_zdw_sum
        print("updating Pn(Zk|Di)...\r")
        for k in range(self.n_topic):
            for d in range(self.n_doc):
                _d_dw_zdw = 0
                for j in range(self.n_word):
                    _d_dw_zdw += self.corpus[d][j] * self.p_z_dw[k][d][j]
                _d_dw_zdw_sum = 0
                for kk in range(self.n_topic):
                    _d_dw_zdw_i = 0
                    for j in range(self.n_word):
                        _d_dw_zdw_i += self.corpus[d][j] * self.p_z_dw[kk][d][j]
                    _d_dw_zdw_sum += _d_dw_zdw_i
                if _d_dw_zdw_sum <= 0:
                    _d_dw_zdw_sum = 1e-6
                self.p_z_d[k][d] = _d_dw_zdw / _d_dw_zdw_sum

    # 计算最大似然值
    def _cal_max_likelihood(self):
        self.likelihood = 0
        for d in range(self.n_doc):
            for j in range(self.n_word):
                _dP_wjdi = 0
                for k in range(self.n_topic):
                    _dP_wjdi += self.p_w_z[j][k] * self.p_z_d[k][d]
                _dP_wjdi = 1.0 / self.n_doc * _dP_wjdi
                self.likelihood += self.corpus[d][j] * math.log(_dP_wjdi)

    # 迭代训练
    def train(self, n_iter=100, d_delta=1e-6, log_fn="log.log"):
        itr = 0
        delta = 10e9
        _likelihood = 0
        f = open(log_fn, 'w')
        while itr < n_iter and delta > d_delta:
            _likelihood = self.likelihood
            self._e_step()
            self._m_step()
            self._cal_max_likelihood()
            itr += 1
            delta = abs(self.likelihood - _likelihood)
            t1 = datetime.datetime.now().strftime('%Y-%m-%d-%y %H:%M:%S');
            f.write("%s iteration %d, max-likelihood = %.6f\n" % (t1, itr, self.likelihood))
            print("%s iteration %d, max-likelihood = %.6f" % (t1, itr, self.likelihood))
        f.close()

    def printVocabulary(self):
        print("vocabulary:")
        for word in self.vocab:
            print(word)

    def saveVocabulary(self, fn):
        f = codecs.open(fn, 'w', 'utf-8')
        for word in self.vocab:
            f.write("%s\n" % word)
        f.close()

    def printWordOfTopic(self):
        for k in range(self.n_topic):
            print("Topic %d" % k)
            for j in range(self.n_word):
                print(self.p_w_z[j][k])

    def saveWordOfTopic(self, fn):
        f = open(fn, 'w', encoding='utf-8')
        for j in range(self.n_word):
            f.write(", w%d" % j)
        f.write("\n")
        for k in range(self.n_topic):
            f.write("topic %d" % k)
            for j in range(self.n_word):
                f.write(", %.6f" % self.p_w_z[j][k])
            f.write("\n")
        f.close()

    def printTopicOfDoc(self):
        for d in range(self.n_doc):
            print("Doc %d" % d)
            for k in range(self.n_topic):
                print(self.p_z_d[k][d])

    def saveTopicOfDoc(self, fn):
        f = open(fn, 'w', encoding='utf-8')
        for k in range(self.n_topic):
            f.write(", z%d" % k)
        f.write("\n")
        for d in range(self.n_doc):
            f.write("doc %d" % d)
            for k in range(self.n_topic):
                f.write(", %.6f" % self.p_z_d[k][d])
            f.write("\n")
        f.close()


if __name__ == "__main__":
    _plsa = pLSA(5)
    _plsa.loadCorpus("./data/corpus.txt")
    _plsa.train()
    _plsa.printTopicOfDoc()
    _plsa.printWordOfTopic()
    _plsa.saveTopicOfDoc("./model/plsa_topic_doc.txt")
    _plsa.saveWordOfTopic("./model/plsa_word_topic.txt")
