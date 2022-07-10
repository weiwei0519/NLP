# coding=UTF-8
# NLTK Natural Language Toolkit

import nltk
from nltk.text import Text
import jieba
from nltk.corpus import sinica_treebank

raw = open('data/sd12.txt', encoding='utf8').read()
text = Text(jieba.lcut(raw))
print(text)

print(sinica_treebank.words())
sinica_text = nltk.Text(sinica_treebank.words())
sinica_text.concordance('我')  # 获取输入词的上下文
print(sinica_text)

# 这个方法用来计算字词在语料库中出现的频率
sinica_fd = nltk.FreqDist(sinica_treebank.words())
top100 = sinica_fd.items()
i = 0
for (x, y) in top100:
    print(x, y)
    if i > 100:
        break
    else:
        i += 1
