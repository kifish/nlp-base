import re
import os
import json
from tqdm import tqdm
from collections import Counter, defaultdict
import numpy as np
import tensorflow as tf


pure_texts = []
pure_tags = []
stops = u'，。！？；、：,\.!\?;:\n'
for line in tqdm(open('../data/msr_train.txt',encoding='utf8')):
    line = [i.strip(' ') for i in re.split('['+stops+']', line) if i.strip(' ')]
    for t in line:
        pure_texts.append('')
        pure_tags.append('')
        for w in re.split(' +', t):
            pure_texts[-1] += w
            if len(w) == 1:
                pure_tags[-1] += 's'
            else:
                pure_tags[-1] += 'b' + 'm'*(len(w)-2) + 'e'


ls = [len(i) for i in pure_texts]
ls = np.argsort(ls)[::-1]
pure_texts = [pure_texts[i] for i in ls]
pure_tags = [pure_tags[i] for i in ls]

min_count = 2
word_count = Counter(''.join(pure_texts))
word_count = Counter({i:j for i,j in word_count.items() if j >= min_count})
word2id = defaultdict(int)
id_here = 0
for i in word_count.most_common():
    id_here += 1
    word2id[i[0]] = id_here

# json.dump(word2id, open('word2id.json', 'w'))





# 未知的词的id为0
vocabulary_size = len(word2id) + 1
tag2vec = {'s':[1, 0, 0, 0], 'b':[0, 1, 0, 0], 'm':[0, 0, 1, 0], 'e':[0, 0, 0, 1]}




batch_size = 1024

def data():
    l = len(pure_texts[0])
    x = []
    y = []
    for i in range(len(pure_texts)):
        if len(pure_texts[i]) != l or len(x) == batch_size:
            yield x,y
            x = []
            y = []
            l = len(pure_texts[i])
        #一行是一个list
        x.append([word2id[j] for j in pure_texts[i]])
        y.append([tag2vec[j] for j in pure_tags[i]])



if not os.path.exists('./tmp/'):
    embedding_size = 128
    keep_prob = tf.placeholder(tf.float32,name='keep_prob')
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    x = tf.placeholder(tf.int32, shape=[None, None],name = 'x')
    embedded = tf.nn.embedding_lookup(embeddings, x)
    # 可参考https://github.com/kifish/cnn-text-classification-tf/blob/master/text_cnn.py#L36
    # embedded [None,seq_len,embedding_size]
    # embedding_size 相当于图像里的in_channels,即词向量相当于图像中的RGB
    embedded_dropout = tf.nn.dropout(embedded, keep_prob)
    W_conv1 = tf.Variable(tf.random_uniform([3, embedding_size, embedding_size], -1.0, 1.0))
    b_conv1 = tf.Variable(tf.random_uniform([embedding_size], -1.0, 1.0))
    # https://github.com/kifish/cnn-text-classification-tf/blob/master/text_cnn.py#L50
    # W_conv1 :  [filter_width, in_channels, out_channels] 即 [行数,列宽(词向量的维度),卷积个数]
    # 可以这么理解,相乘的时候是这样对齐的:
    # embedded_dropout  [None,seq_len,embedding_size]
    #                            |             |          卷积核的个数
    # W_conv1                [filter_width, in_channels, out_channels]
    # 一个样本
    # conv1d之后 [seq_len,in_channels,out_channels]
    # + b_conv1

    # 对于的一个卷积核,词向量的维度上都加了一个同样的数
    # 不同的卷积核,词向量的维度上加的数不一样

    y_conv1 = tf.nn.relu(tf.nn.conv1d(embedded_dropout, W_conv1, stride=1, padding='SAME') + b_conv1)
    W_conv2 = tf.Variable(tf.random_uniform([3, embedding_size, int(embedding_size/4)], -1.0, 1.0))
    b_conv2 = tf.Variable(tf.random_uniform([int(embedding_size/4)], -1.0, 1.0)) # 长度 对应了卷积核的个数
    y_conv2 = tf.nn.relu(tf.nn.conv1d(y_conv1, W_conv2, stride=1, padding='SAME') + b_conv2)
    W_conv3 = tf.Variable(tf.random_uniform([3, int(embedding_size/4), 4], -1.0, 1.0))
    b_conv3 = tf.Variable(tf.random_uniform([4], -1.0, 1.0))
    pred = tf.nn.softmax(tf.nn.conv1d(y_conv2, W_conv3, stride=1, padding='SAME') + b_conv3,name = 'pred')
    label = tf.placeholder(tf.float32, shape=[None, None, 4])
    cross_entropy = - tf.reduce_sum(label * tf.log(pred + 1e-20))
    train_step = tf.train.AdamOptimizer().minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(pred, 2), tf.argmax(label, 2))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    # nb_epoch = 300
    nb_epoch = 15
    for i in range(nb_epoch):
        d = tqdm(data(), desc='Epcho %s, Accuracy: 0.0' % (i+1))
        k = 0
        accs = []
        for xxx, yyy in d:
            k += 1
            if k % 100 == 0:
                acc = sess.run(accuracy, feed_dict={x: xxx, label: yyy, keep_prob:1})
                accs.append(acc)
                d.set_description('Epcho %s, Accuracy: %s'%(i+1, acc))
            sess.run(train_step, feed_dict={x: xxx, label: yyy, keep_prob:0.3})
        print('\nEpcho %s Mean Accuracy: %s'%(i+1, np.mean(accs)))

    saver = tf.train.Saver()
    saver.save(sess, './tmp/cnn_model')

else:
    sess = tf.Session()
    saver = tf.train.import_meta_graph("./tmp/cnn_model.meta")
    saver.restore(sess, "./tmp/cnn_model")
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    pred = graph.get_tensor_by_name("pred:0")
    keep_prob = graph.get_tensor_by_name("keep_prob:0")



add_dict = {}
if os.path.exists('add_dict.txt'):
    with open('add_dict.txt',encoding='utf8') as f:
        for l in f:
            a, b = l.split(',')
            add_dict[a] = np.log(float(b))

from pkg.hmm.viterbi import viterbi
def simple_cut(s):
    if s:
        # 这里np.log(0) = -inf
        # np.log(k+0.00001) 直接这样平移效果不好

        # nodes = [dict(zip('sbme', np.log(k+0.00001)))
        #          for k in sess.run(pred, feed_dict={x: [[word2id[i] for i in s]], keep_prob: 1})[0]
        #          ]

        nodes = []
        for k in sess.run(pred, feed_dict={x: [[word2id[i] for i in s]], keep_prob: 1})[0]:
            for idx,num in enumerate(k):
                if num <= 0.0000000001:
                    k[idx] = 0.0000000001
            nodes.append(dict(zip('sbme', np.log(k))))

        for w, f in add_dict.items():
            for i in re.finditer(w, s):
                if len(w) == 1:
                    nodes[i.start()]['s'] += f
                else:
                    nodes[i.start()]['b'] += f
                    nodes[i.end() - 1]['e'] += f
                    for j in range(i.start() + 1, i.end() - 1):
                        nodes[j]['m'] += f
        tags = viterbi(nodes)
        words = [s[0]]
        for i in range(1, len(s)):
            if tags[i] in ['s', 'b']:
                words.append(s[i])
            else:
                words[-1] += s[i]
        return words
    else:
        return []


def cut_words(s):
    i = 0
    r = []
    for j in re.finditer('[' + stops + ' ]' + '|[a-zA-Z\d]+', s):
        r.extend(simple_cut(s[i:j.start()]))
        r.append(s[j.start():j.end()])
        i = j.end()
    if i != len(s):
        r.extend(simple_cut(s[i:]))
    return r



print(cut_words('扫描二维码,关注微信号'))

if __name__ == '__main__':
    while True:
        s = input('请输入\n')
        print(cut_words(s))




# 2014人民日报语料 预处理

"""
import glob
import re
from tqdm import tqdm
from collections import Counter, defaultdict
import json
import numpy as np
import os

txt_names = glob.glob('./2014/*/*.txt')

pure_texts = []
pure_tags = []
stops = u'，。！？；、：,\.!\?;:\n'
for name in tqdm(iter(txt_names)):
    txt = open(name).read().decode('utf-8', 'ignore')
    txt = re.sub('/[a-z\d]*|\[|\]', '', txt)
    txt = [i.strip(' ') for i in re.split('['+stops+']', txt) if i.strip(' ')]
    for t in txt:
        pure_texts.append('')
        pure_tags.append('')
        for w in re.split(' +', t):
            pure_texts[-1] += w
            if len(w) == 1:
                pure_tags[-1] += 's'
            else:
                pure_tags[-1] += 'b' + 'm'*(len(w)-2) + 'e'

"""
