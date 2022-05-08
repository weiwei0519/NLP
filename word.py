# coding=UTF-8
# 词分析

import jieba
import jieba.posseg as psg
from hmm import HMM
import re
import datetime


# 分词，目前用jieba分词工具
# # 全模式分词，把句子中所有可以成词的词语都扫描出来，词语会重复，且不能解决歧义，适合关键词提取
# words = jieba.cut(sent, cut_all=True)
# # 精确模式分词，将句子最精确的切分，此为默认模式，适合文本分析
# # 默认模式调用，可以忽略cut_all参数，写法如下：
# # seg_list = jieba.cut(sent)
# seg_list = jieba.cut(sent, cut_all=False)
# # 搜索引擎模式，在精确模式的基础上，对长词再次切分，提高召回率，适合用于搜索引擎分词
# seg_list = jieba.cut_for_search(sent)
def cut(text, cuttype=False):
    if text == "":
        return
    if cuttype:
        words = list(jieba.cut(text, cut_all=True))
    else:
        words = list(jieba.cut(text, cut_all=False))
    return words


# 分词，且过滤停用词
def cut_rm_stopwords(text, cuttype=False):
    words = filter_stopwords(cut(text, cuttype))
    return words


# 加载停用词词典
def load_stopwords(path='./dictionary/stop_words.utf8'):
    with open(path, 'r', encoding='UTF-8', errors='ignore') as f:
        return [line.strip() for line in f]


# 过滤停用词
def filter_stopwords(words):
    stopwords = load_stopwords()
    if words[0].find('/') > 0:
        words = [x for x in words if x[:x.find('/')] not in stopwords]
    else:
        words = [x for x in words if x not in stopwords]
    return words

# 分词且带词性（pos：part of speech）
def cut_with_pos(text):
    words_pos = psg.cut(text)
    words_pos = ['{0}/{1}'.format(w, pos) for w, pos in words_pos]
    return words_pos

# 分词且带词性（pos：part of speech），并过滤停用词
def cut_with_pos_rm_sw(text):
    words_pos = psg.cut(text)
    words_pos = ['{0}/{1}'.format(w, pos) for w, pos in words_pos]
    stopwords = load_stopwords()
    words_pos = [x for x in words_pos if x[:x.find('/')] not in stopwords]
    return words_pos


# 计算text中词的词频，且默认会过滤停用词，因为绝大多数场景，停用词无实际意义
# sort定义是否需要排序输出
def calc_TF(text, sort=False):
    words = cut_rm_stopwords(text)
    words_tf = {}
    for w in words:
        words_tf[w] = words_tf.get(w, 0) + 1
    if sort:
        return sorted(words_tf.items(), key=lambda x: x[1], reverse=True)
    else:
        return words_tf


def load_dictionary(path='./dictionary/dictionary.txt'):
    # print("load dictionary start_time: {0}".format(datetime.datetime.now()))
    dic = []
    with open(path, 'r', encoding='UTF-8', errors='ignore') as f:
        for line in f:
            dic.extend(word for word in line.strip().split(" "))
    print("loaded dictionary has {0} words.".format(len(dic)))
    # print("load dictionary end_time: {0}".format(datetime.datetime.now()))
    return dic


# 正向最大匹配分词算法
def cut_mm(test):
    result = []
    index = 0
    text_len = len(text)
    window_size = 3
    dic = load_dictionary()
    while text_len > index:
        for size in range(window_size + index, index, - 1):  # 3,2,1,0
            piece = text[index:size]
            if piece in dic:
                index = size - 1
                result.append(piece)
                break
        index = index + 1

    print(result)
    return result


# 逆向最大匹配分词算法
def cut_rmm(text):
    result = []
    index = len(text)
    window_size = 3
    dic = load_dictionary()
    while index > 0:
        for size in range(index - window_size, index):  # 3,2,1,0
            piece = text[size:index]
            if piece in dic:
                index = size + 1
                result.append(piece)
                break
        index = index - 1
    result.reverse()
    print(result)
    return result


# 双向最大匹配算法
def cut_bdmm(text):
    result = []
    mmSegRst = cut_mm(text)
    rmmSegRst = cut_rmm(text)
    if len(mmSegRst) < len(rmmSegRst):
        result = mmSegRst
    elif len(mmSegRst) > len(rmmSegRst):
        result = rmmSegRst
    else:
        mmSegRst_s = [x for x in mmSegRst if len(x) == 1]
        rmmSegRst_s = [y for y in rmmSegRst if len(y) == 1]
        if len(mmSegRst_s) < len(rmmSegRst_s):
            result = mmSegRst
        else:
            result = rmmSegRst
    print(result)
    return result

# 判断是否包含可分词的汉字
def ishanzi(text):
    re_han_internal = re.compile("([\u4e00-\u9FD5a-zA-Z0-9+#&\._]+)")
    return re_han_internal.search(text)


if __name__ == '__main__':
    text = "我想学习<机器学习第2部>"
    print("input text: {0}".format(text))
    words = cut(text)
    print("jieba精确分词结果：{0}".format(words))
    words = cut_rm_stopwords(text)
    print("jieba分词且过滤停用词结果：{0}".format(words))
    words_tf = calc_TF(text)
    print("jieba分词并计算词频结果：{0}".format(words_tf))
    words_pos = cut_with_pos(text)
    print("jieba.posseg分词且带词性结果：{0}".format(words_pos))
    words_pos = cut_with_pos_rm_sw(text)
    print("jieba.posseg分词且带词性并过滤停用词结果：{0}".format(words_pos))
    # 隐马尔可夫模型分词
    hmm = HMM(model_file='./model/hmm_pos_model.pkl', dic_file='./dictionary/dic_pos.txt', need_pos=True)
    words_hmm = list(hmm.cut(text))
    print("HMM分词结果：{0}".format(words_hmm))
    words_hmm_rm_sw = filter_stopwords(words_hmm)
    print("HMM分词过滤停用词结果：{0}".format(words_hmm_rm_sw))
