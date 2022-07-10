# coding=UTF-8
# 句法分析

# PCFG句法分析
from nltk.parse import stanford
from nltk.parse.corenlp import CoreNLPParser
from nltk import word_tokenize
from word import *
import os


class PCFG():
    def __init__(self):
        self.parser_path = './Stanford_Parser/stanford-parser.jar'
        self.model_path = './Stanford_Parser/stanford-parser-4.2.0-models.jar'
        # PCFG模型路径, 此路径为model_path路径下的文件。
        self.pcfg_path = 'edu/stanford/nlp/models/lexparser/chinesePCFG.ser.gz'
        # 指定JDK路径
        if not os.environ.get('JAVA_HOME'):
            JAVA_HOME = 'C:\Program Files\Java\jdk1.8.0_181'
            os.environ['JAVA_HOME'] = JAVA_HOME
        self.parser = stanford.StanfordParser(path_to_jar=self.parser_path,
                                              path_to_models_jar=self.model_path,
                                              model_path=self.pcfg_path)
        self.coreNLPParser = CoreNLPParser(r'')

    def stanford_parser(self, sentence):
        word_list = cut_org(sentence)
        word_list = ' '.join(word_list)  # stanford parser只能接受空格分隔的词列表
        parser_result = self.parser.raw_parse(word_list)
        for line in parser_result:
            print(line.leaves())
            line.draw()
        return parser_result

    def coreNLP_parser(self, sentence):
        # 运行前，先启动Stanford CoreNLP server
        # java -Xmx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -serverProperties StanfordCoreNLP-chinese.properties -port 9000 -timeout 15000
        word_list = cut_org(sentence)
        word_list = ' '.join(word_list)
        coreNLP_parser = CoreNLPParser(url='http://localhost:9000')
        result = self.coreNLP_parser(word_list)
        return result

if __name__ == '__main__':
    sentence = '他骑自行车去了菜市场。'
    pcfg = PCFG()
    result = pcfg.stanford_parser(sentence)
    for line in result:
        print(line)
    result = pcfg.coreNLP_parser(sentence)
    print(result)
