# coding=UTF-8
# 字典操作的工具类

class dictionary():

    def __init__(self, dic_file='./dictionary/dic_pos.txt', Task='CUT'):

        # 词典文件
        self.dic_file = dic_file

        # 区分不同的训练类型：POS，Normal, NER
        self.Task = Task

        # 从分词字典加载的观察数据集和状态数据集
        self.Z, self.Words, self.CUT_state, self.POS_state = self.read_data_from_dic()

        self.Z_nums = sum([len(Zm) for Zm in self.Z])

    # 给输入word计算序列，并附上pos
    def buildState(self, word, pos, Task):
        labels = []
        if Task == 'POS':
            if len(word) == 1:
                labels.append('S_' + pos)
            else:
                labels += ['B_' + pos] + ['M_' + pos] * (len(word) - 2) + ['E_' + pos]
        elif Task == 'CUT':
            if len(word) == 1:
                labels.append('S')
            else:
                labels += ['B'] + ['M'] * (len(word) - 2) + ['E']
        elif Task == 'NER':
            if type == 'NER':
                if len(word) == 1:
                    labels.append('S_' + pos)
                else:
                    labels += ['B_' + pos] + ['M_' + pos] * (len(word) - 2) + ['E_' + pos]
            else:
                # 如果当前任务类型为NER，但是外部传入了Normal，说明从字典看是非NER词，标记为O
                labels += ['O_' + pos] * (len(word))
        return labels

    def read_data_from_dic(self):
        # 考虑到分词、词性标注、NER标注在CRF中可以设定为流水线任务，所以加载dictionary的时候，一次性取出。
        Z = list()  # 汉字列表
        Words = list()  # 词语列表
        NER_word = list()  # 命名识别词语列表
        CUT_state = list()  # 分词状态列表
        POS_state = list()  # 词性状态列表
        NER_state = list()  # NER词性状态列表
        print("start read dictionary data")
        line_num = 0
        with open(self.dic_file, encoding='utf-8') as f:  # 读取字典文件
            for line in f:
                line_num += 1
                line = line.strip()  # 移除字符串头尾指定的字符（默认为空格或换行符）或字符序列
                if not line: continue
                split_str = line.split(' ')  # 将空格分隔的提取出
                z = []
                word = []
                words = []
                cut_state = []
                pos_state = []
                ner_words = []
                ner_state = []
                ner_word = ''
                ner_switch = False
                for str in split_str:  # 国务院/nt [西藏/ns 自治区/n 政府/n]nt
                    if str.find('[') >= 0:
                        # ner_switch = True
                        # ner_word = str[str.find('[') + 1:str.find('/')]
                        str = str[str.find('[') + 1:-1]
                    if str.find(']') > 0:
                        # ner_switch = False
                        # ner_word += str[:str.find('/')]
                        str = str[:str.find(']')]
                    word = str[:str.find('/')]
                    words.append(word)
                    # pos_state.append(str[str.find('/') + 1:])
                    pos = str[str.find('/') + 1:]
                    z.extend(i for i in word)
                    cut_state.extend(self.buildState(word, pos, Task='CUT'))
                    pos_state.extend(self.buildState(word, pos, Task='POS'))
                Z.append(z)
                CUT_state.append(cut_state)
                Words.append(words)
                POS_state.append(pos_state)
        print("finish read dictionary data")
        f.close()
        return Z, Words, CUT_state, POS_state
