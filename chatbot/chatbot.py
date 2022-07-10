# coding=UTF-8
# 基于tensorflow2 seq2seq + keras 搭建聊天机器人

'''
File Name: chatbot.py
Program IDE: PyCharm
Created Time: 2022/6/5 0005 16:18
Author: Wei Wei
'''

from word import *
import tensorflow as tf
import keras
from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense
import os
import numpy as np

encoder_file = './corpus_data/Q.txt'
decoder_file = './corpus_data/A.txt'
encoder_word_list = []
encoder_vocabulary = set()
en_vocab2id = dict()
en_id2vocab = dict()
decoder_word_list = []
decoder_vocabulary = set()
de_vocab2id = dict()
de_id2vocab = dict()

start_word = '--'
stop_word = '\n'

# 语料库预处理：
# encoder序列处理，生成encoder词序列，词典
with open(encoder_file, 'r', encoding='utf8') as en:
    for sentence in en.readlines():
        word_list = [start_word]
        word_list.extend(cut_org(sentence.strip()))
        word_list.append(stop_word)
        encoder_word_list.append(word_list)
        for word in word_list:
            if word not in encoder_vocabulary:
                encoder_vocabulary.add(word)
                id = len(en_vocab2id)
                en_vocab2id[word] = id
                en_id2vocab[id] = word
    en.close()
# decoder序列处理，生成decoder词序列，词典
with open(decoder_file, 'r', encoding='utf8') as de:
    for sentence in de.readlines():
        word_list = [start_word]
        word_list.extend(cut_org(sentence.strip()))
        word_list.append(stop_word)
        decoder_word_list.append(word_list)
        for word in word_list:
            if word not in decoder_vocabulary:
                decoder_vocabulary.add(word)
                id = len(de_vocab2id)
                de_vocab2id[word] = id
                de_id2vocab[id] = word
    de.close()

max_training_samples = min(25000, len(encoder_word_list) - 1)  # 定义了训练使用的行数

# 建立字符字典
# 将字符集装换为排序后的字符列表
encoder_vocabulary = sorted(encoder_vocabulary)
decoder_vocabulary = sorted(decoder_vocabulary)

encoder_vocab_size = len(encoder_vocabulary)
decoder_vocab_size = len(decoder_vocabulary)
max_encoder_seq_length = max([len(w) for w in encoder_word_list])
max_decoder_seq_length = max([len(w) for w in decoder_word_list])
# 生成独热码训练集
encoder_input_data = np.zeros((len(encoder_word_list), max_encoder_seq_length, encoder_vocab_size), dtype='float32');
decoder_input_data = np.zeros((len(encoder_word_list), max_decoder_seq_length, decoder_vocab_size), dtype='float32');
decoder_target_data = np.zeros((len(encoder_word_list), max_decoder_seq_length, decoder_vocab_size), dtype='float32');
# 将每个时刻字符索引设置为1
for i, (encoder_word, decoder_word) in enumerate(
        zip(encoder_word_list, decoder_word_list)):
    for t, w in enumerate(encoder_word):
        encoder_input_data[i, t, en_vocab2id[w]] = 1.
    for t, w in enumerate(decoder_word):
        decoder_target_data[i, t, de_vocab2id[w]] = 1.
        if t > 0:
            decoder_input_data[i, t - 1, de_vocab2id[w]] = 1
# 训练序列到序列聊天机器人
batch_size = 64
epochs = 50
num_neurons = 256
encoder_inputs = Input(shape=(None, encoder_vocab_size))
encoder_lstm = LSTM(num_neurons, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
encoder_states = [state_h, state_c]
decoder_inputs = Input(shape=(None, decoder_vocab_size))
decoder_lstm = LSTM(num_neurons, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(decoder_vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
cp_callback = keras.callbacks.ModelCheckpoint(filepath='模型参数.h5',
                                              save_weights_only=False, save_best_only=True)
model.fit([encoder_input_data, decoder_input_data],
          decoder_target_data, batch_size=batch_size, epochs=epochs,
          validation_split=0.1, callbacks=[cp_callback])
# 组装序列生成模型
encoder_model = Model(encoder_inputs, encoder_states)
thought_input = [Input(shape=(num_neurons,)), Input(shape=(num_neurons,))]
decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=thought_input)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(inputs=[decoder_inputs] + thought_input,
                      outputs=[decoder_outputs] + decoder_states)


# 预测输出序列
def decode_sequence(input_seq):
    thought = encoder_model.predict(input_seq)  # 生成思想向量作为解码器的输入
    target_seq = np.zeros((1, 1, decoder_vocab_size))  # 与训练相反，target_seq一开始是一个零张量
    target_seq[0, 0, de_vocab2id[stop_word]] = 1.
    stop_condition = False
    generated_sequence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + thought)  # 将已生成的词条和最新的状态传递给解码器，以预测下一个序列元素
        generated_token_idx = np.argmax(output_tokens[0, -1:1])
        generated_char = de_id2vocab[generated_token_idx]
        generated_sequence += generated_char
        if (generated_char == stop_word or len(generated_sequence) > max_decoder_seq_length):
            stop_condition = True
        target_seq = np.zeros((1, 1, decoder_vocab_size))  # 更新目标序列，并使用最后生成的词条作为下一生成步骤的输入
        target_seq[0, 0, generated_token_idx] = 1.
        thought = [h, c]  # 更新思想状态
    return generated_sequence


# 生成回复
def response(input_text):
    input_seq = np.zeros((1, max_encoder_seq_length, encoder_vocab_size), dtype='float32')
    for t, char in enumerate(input_text):  # 对输入文本的每个字符进行循环遍历
        input_seq[0, t, en_vocab2id[char]] = 1.
    decoded_sentence = decode_sequence(input_seq)
    print('Bot Reply(Decoded sentence):', decoded_sentence)


response(cut_org("--新股申购如何操作？"))
