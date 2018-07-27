import sys
import gensim
import sklearn
import os
import numpy as np
import jieba
from gensim.models.doc2vec import Doc2Vec, LabeledSentence

#下面这一行写的不太好，可以考虑直接import *
TaggededDocument = gensim.models.doc2vec.TaggedDocument
def get_datasest():
    if not os.path.exists('../data/wangyi_title_cut.txt'):
        #中文分词
        jieba.enable_parallel()
        #一行代表一个标题
        line_num  = 0
        with open('../data/wangyi_title_cut.txt','w') as fw:
            #没处理标点符号
            with open("../data/wangyi_title.txt", 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    line = line.replace("\r\n","")
                    line = line.replace(" ","")
                    line_seg = jieba.cut(line) #list
                    line_seg = " ".join(line_seg)
                    line_num += 1
                    fw.write(line_seg+'\n')

        print('setence_num in raw corpus:',line_num)

    with open("../data/wangyi_title_cut.txt", 'r') as f:
        corpus = f.readlines()
        print('setence_num in corpus_seg',len(corpus))

    x_train = []
    #y = np.concatenate(np.ones(len(docs)))
    for idx, text in enumerate(corpus):
        word_list = text.strip().split(' ')
        document = TaggededDocument(word_list, tags=[idx])
        x_train.append(document)

    return x_train

#取出documents对应的向量
def getVecs(model, corpus, size):
    vecs = [np.array(model.docvecs[z.tags[0]].reshape(1, size)) for z in corpus]
    return np.concatenate(vecs)

def train(x_train, size=200, epoch_num=1):
    model_dm = Doc2Vec(x_train,min_count=1, window = 3, size = size, sample=1e-3, negative=5, workers=4)
    model_dm.train(x_train, total_examples=model_dm.corpus_count, epochs=70)
    model_dm.save('./model_dm_wangyi')

    return model_dm

def test():
    model_dm = Doc2Vec.load("./model_dm_wangyi")
    test_text = ['《', '舞林', '争霸' '》', '十强' '出炉', '复活', '舞者', '澳门', '踢馆']
    inferred_vector_dm = model_dm.infer_vector(test_text)
    #把text转化为向量
    print(inferred_vector_dm)
    sims = model_dm.docvecs.most_similar([inferred_vector_dm], topn=10)

    return sims

if __name__ == '__main__':
    x_train = get_datasest()

    model_dm = train(x_train)

    sims = test()
    for idx, sim in sims:   #字典，通过idx可以找到对应的document
        sentence = x_train[idx]
        words = ''
        for word in sentence[0]:
            words += word + ' '
        print('sentence_seg',words,'sim_val:',sim)

'''
vec : ...
sentence_seg 《 舞林 争霸 》 十强 出炉 复活 舞者 澳门 踢馆  sim_val: 0.46345528960227966
sentence_seg MJ 环球 春晚 “ 复活 ” 全场 尖叫 林俊杰 再现 经典  sim_val: 0.45896923542022705
sentence_seg 《 舞 出 》 撒 贝宁 跳 “ 苦情 舞 ” 复活 陈冲 再现 经典  sim_val: 0.4131377637386322
sentence_seg 春晚 语言 类节目 今天 通排 姜昆 相声 可能 “ 复活 ”  sim_val: 0.36322033405303955
sentence_seg 京华 时报 ： 金曲奖 黯淡 中生代 复活  sim_val: 0.3371550738811493
sentence_seg 《 我 是 歌手 》 尚雯婕 出局 终极 替补 彭佳慧 献声  sim_val: 0.3297603726387024
sentence_seg 虎年 春晚 看点 揭秘 黄宏 复活 《 整容 》 一次 过关  sim_val: 0.32491597533226013
sentence_seg 威尼斯 电影节 临时 加场 中国 媒体 见面会  sim_val: 0.3171465992927551
sentence_seg " 好 声音 " 终极 考核 首场 吉克隽逸 晋级 徐海 星 出局  sim_val: 0.31404680013656616
sentence_seg 《 星 跳水 立方 》 台湾 馆 启动 吴建豪 勇跳 十米 台  sim_val: 0.31241583824157715
'''
