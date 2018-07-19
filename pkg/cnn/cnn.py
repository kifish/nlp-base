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

json.dump(word2id, open('word2id.json', 'w'))
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
        x.append([word2id[j] for j in pure_texts[i]])
        y.append([tag2vec[j] for j in pure_tags[i]])




embedding_size = 128
keep_prob = tf.placeholder(tf.float32)

embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
x = tf.placeholder(tf.int32, shape=[None, None])
embedded = tf.nn.embedding_lookup(embeddings, x)
embedded_dropout = tf.nn.dropout(embedded, keep_prob)
W_conv1 = tf.Variable(tf.random_uniform([3, embedding_size, embedding_size], -1.0, 1.0))
b_conv1 = tf.Variable(tf.random_uniform([embedding_size], -1.0, 1.0))
y_conv1 = tf.nn.relu(tf.nn.conv1d(embedded_dropout, W_conv1, stride=1, padding='SAME') + b_conv1)
W_conv2 = tf.Variable(tf.random_uniform([3, embedding_size, int(embedding_size/4)], -1.0, 1.0))
b_conv2 = tf.Variable(tf.random_uniform([int(embedding_size/4)], -1.0, 1.0))
y_conv2 = tf.nn.relu(tf.nn.conv1d(y_conv1, W_conv2, stride=1, padding='SAME') + b_conv2)
W_conv3 = tf.Variable(tf.random_uniform([3, int(embedding_size/4), 4], -1.0, 1.0))
b_conv3 = tf.Variable(tf.random_uniform([4], -1.0, 1.0))
y = tf.nn.softmax(tf.nn.conv1d(y_conv2, W_conv3, stride=1, padding='SAME') + b_conv3)

y_ = tf.placeholder(tf.float32, shape=[None, None, 4])
cross_entropy = - tf.reduce_sum(y_ * tf.log(y + 1e-20))
train_step = tf.train.AdamOptimizer().minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 2), tf.argmax(y_, 2))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
# nb_epoch = 300
nb_epoch = 10
for i in range(nb_epoch):
    d = tqdm(data(), desc=u'Epcho %s, Accuracy: 0.0'%(i+1))
    k = 0
    accs = []
    for xxx,yyy in d:
        k += 1
        if k%100 == 0:
            acc = sess.run(accuracy, feed_dict={x: xxx, y_: yyy, keep_prob:1})
            accs.append(acc)
            d.set_description('Epcho %s, Accuracy: %s'%(i+1, acc))
        sess.run(train_step, feed_dict={x: xxx, y_: yyy, keep_prob:0.5})
    print('Epcho %s Mean Accuracy: %s'%(i+1, np.mean(accs)))

saver = tf.train.Saver()
saver.save(sess, './tmp/cw.ckpt')






''

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
