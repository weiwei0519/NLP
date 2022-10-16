# coding=UTF-8
# 词向量模型，基于Gensim
'''
File Name: word2vec.py
Program IDE: PyCharm
Created Time: 2022/6/4 0004 16:52
Author: Wei Wei
'''

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import logging
from word import *
import glob

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class WordVector():
    def __init__(self, corpus_file='./datasets/Chinese/*.txt', model_file='./model/word2vec_model'):
        self.corpus_word_list = self.read_corpus_file(corpus_file)
        self.wf, self.id2word, self.word2id = self.word_process()
        self.model_file = model_file
        self.train()

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

    def train(self):
        # 加载模型
        try:
            self.model = Word2Vec.load(self.model_file)
            if self.model != None:
                return
            else:
                raise FileNotFoundError
        except FileNotFoundError:
            print("NNLM model file doesn't exist. Need train.")
        # sg=0 CBOW模型；sg=1 Skip-gram模型
        # windows n-gram选词的窗口
        # min_count 表示最小出现次数，如果一个词出现的次数小于该值，那么直接忽略该词语。
        # workers 表示训练的线程数
        self.model = Word2Vec(self.corpus_word_list, sg=0, window=3, min_count=1, workers=8)
        self.model.save(self.model_file)


if __name__ == '__main__':
    word2vec = WordVector()

    print("'北京'与'上海'词语相似度为：" + str(word2vec.model.wv.similarity('北京', '上海')))  # 相似度为0.63
    print("'大学'与'学院'词语相似度为：" + str(word2vec.model.wv.similarity('大学', '学院')))  # 相似度为0.44
    word = '中国'
    if word in word2vec.model.wv.index_to_key:
        print('与', word, '相似的词有：')
        print(word2vec.model.wv.most_similar(word))

    cont = True
    while cont:
        sent = str(input("请输入你想查找相似词语的词： "))

        if sent == "exit":
            cont = False
        else:
            if sent in word2vec.model.wv.index_to_key:
                print('与', sent, '相似的词有：')
                print(word2vec.model.wv.most_similar(sent))
