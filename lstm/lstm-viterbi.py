import re
import numpy as np
import pandas as pd


# backoff2005语料
# s = open('../msr_train.txt',encoding='gbk').read()
s = open('../msr_train.txt', encoding='utf8').read()
s = s.split('\r\n')

def clean(s): #整理一下数据，有些不规范的地方
    if u'“/s' not in s:
        return s.replace(u' ”/s', '')
    elif u'”/s' not in s:
        return s.replace(u'“/s ', '')
    elif u'‘/s' not in s:
        return s.replace(u' ’/s', '')
    elif u'’/s' not in s:
        return s.replace(u'‘/s ', '')
    else:
        return s

s = u''.join(map(clean, s))
s = re.split(u'[，。！？、]/[bems]', s)

data = [] #生成训练样本
label = []
def get_xy(s):
    s = re.findall('(.)/(.)', s)
    if s:
        s = np.array(s)
        return list(s[:,0]), list(s[:,1])

for i in s:
    x = get_xy(i)
    if x:
        data.append(x[0])
        label.append(x[1])

d = pd.DataFrame(index=range(len(data)))
d['data'] = data
d['label'] = label
maxlen = 32
d = d[d['data'].apply(len) <= maxlen] # 丢掉多于32字的样本
d.index = range(len(d))


chars = [] #统计所有字，跟每个字编号
for i in data:
    chars.extend(i)

chars = pd.Series(chars).value_counts()
chars[:] = range(1, len(chars)+1)

#?  这里没懂

#生成适合模型输入的格式
from keras.utils import np_utils
d['x'] = d['data'].apply(lambda x: np.array(list(chars[x])+[0]*(maxlen-len(x)))) # padding 0
#pandas 真的很慢

tag = pd.Series({'s':0, 'b':1, 'm':2, 'e':3, 'x':4})
def trans_one(x):
    _ = map(lambda y: np_utils.to_categorical(y,5), tag[x].reshape((-1,1)))
    _ = list(_)
    _.extend([np.array([[0,0,0,0,1]])]*(maxlen-len(x)))
    return np.array(_)

# >>> [np.array([[0,0,0,0,1]])]*(2)
# [array([[0, 0, 0, 0, 1]]), array([[0, 0, 0, 0, 1]])]

d['y'] = d['label'].apply(trans_one)

#设计模型
word_size = 128
from keras.layers import Dense, Embedding, LSTM, TimeDistributed, Input, Bidirectional
from keras.models import Model

sequence = Input(shape=(maxlen,), dtype='int32')
embedded = Embedding(len(chars)+1, word_size, input_length=maxlen, mask_zero=True)(sequence)
blstm = Bidirectional(LSTM(64, return_sequences=True), merge_mode='sum')(embedded)
output = TimeDistributed(Dense(5, activation='softmax'))(blstm)
model = Model(input=sequence, output=output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 1024
# history = model.fit(np.array(list(d['x'])), np.array(list(d['y'])).reshape((-1,maxlen,5)), batch_size=batch_size, nb_epoch=50)
model.fit(np.array(list(d['x'])), np.array(list(d['y'])).reshape((-1,maxlen,5)), verbose = 2,batch_size=batch_size, nb_epoch=50)

#最终模型可以输出每个字属于每种标签的概率
#然后用维比特算法来dp

#转移概率，单纯用了等概率
zy = {'be':0.5, 
      'bm':0.5, 
      'eb':0.5, 
      'es':0.5, 
      'me':0.5, 
      'mm':0.5,
      'sb':0.5, 
      'ss':0.5
     }

zy = {i:np.log(zy[i]) for i in zy.keys()}

def viterbi(nodes):
    paths = {'b':nodes[0]['b'], 's':nodes[0]['s']}
    for l in range(1,len(nodes)):
        paths_ = paths.copy()
        paths = {}
        for i in nodes[l].keys():
            nows = {}
            for j in paths_.keys():
                if j[-1]+i in zy.keys():
                    nows[j+i]= paths_[j]+nodes[l][i]+zy[j[-1]+i]
            k = np.argmax(nows.values())
            paths[nows.keys()[k]] = nows.values()[k]
    return paths.keys()[np.argmax(paths.values())]



# >>> i = [0.2,0.2,0.3,0.3,0]
# >>> dict(zip(['s','b','m','e'], i[:4]))
# {'s': 0.2, 'b': 0.2, 'm': 0.3, 'e': 0.3}


def simple_cut(s):
    if s:
        r = model.predict(np.array([list(chars[list(s)].fillna(0).astype(int))+[0]*(maxlen-len(s))]), verbose=False)[0][:len(s)]
        r = np.log(r)
        nodes = [dict(zip(['s','b','m','e'], i[:4])) for i in r]
        t = viterbi(nodes)
        words = []
        for i in range(len(s)):
            if t[i] in ['s', 'b']:
                words.append(s[i])
            else:
                words[-1] += s[i]
        return words
    else:
        return []

not_cuts = re.compile(r'([\da-zA-Z ]+)|[。，、？！\.\?,!]')
def cut_word(s):
    result = []
    j = 0
    for i in not_cuts.finditer(s):
        result.extend(simple_cut(s[j:i.start()]))
        result.append(s[i.start():i.end()])
        j = i.end()
    result.extend(simple_cut(s[j:]))
    return result



print(cut_word('他来到了网易杭研大厦'))
while True:
    input_str = input()
    print(cut_word(input_str))


'''
>>> model.summary()
_______________________________________________________
Layer (type) Output Shape Param # Connected to
=======================================================
input_2 (InputLayer) (None, 32) 0   #None 代表任意数量的样本,32表示一个句子字的个数
_______________________________________________________
embedding_2 (Embedding) (None, 32, 128) 660864 input_2[0][0]  #None 代表任意数量的样本,32表示一个句子字的个数,128表示一个字对应的vec
_______________________________________________________
bidirectional_1 (Bidirectional) (None, 32, 64) 98816 embedding_2[0][0] #None 代表任意数量的样本,32表示一个句子字的个数,64表示一个字对应的vec经过双向LSTM之后提取的feature
_______________________________________________________
timedistributed_2 (TimeDistribute) (None, 32, 5) 325 bidirectional_1[0][0] #  #None 代表任意数量的样本,32表示一个句子字的个数,5表示预测的类别的one hot向量,例如[0,1,0,0,0]
=======================================================
Total params: 760005
'''



'''
# debug模式下训练相当慢,直接run速度稍微快一点
# i5-3570 CPU 3.4GHz
Epoch 1/50
 - 231s - loss: 0.7460 - acc: 0.7048
Epoch 2/50
 - 227s - loss: 0.4109 - acc: 0.8523
Epoch 3/50
 - 227s - loss: 0.3484 - acc: 0.8749
Epoch 4/50
 - 227s - loss: 0.3112 - acc: 0.8888
Epoch 5/50
 - 226s - loss: 0.2848 - acc: 0.8989
Epoch 6/50
 - 41737s - loss: 0.2633 - acc: 0.9070  # 没明白这里为啥消耗了41737s
Epoch 7/50
 - 237s - loss: 0.2450 - acc: 0.9136
Epoch 8/50
 - 233s - loss: 0.2292 - acc: 0.9194
Epoch 9/50
 - 232s - loss: 0.2158 - acc: 0.9244
Epoch 10/50
 - 232s - loss: 0.2041 - acc: 0.9285
Epoch 11/50
 - 228s - loss: 0.1939 - acc: 0.9325
Epoch 12/50
 - 228s - loss: 0.1843 - acc: 0.9360
Epoch 13/50
 - 229s - loss: 0.1760 - acc: 0.9389
Epoch 14/50
 - 229s - loss: 0.1683 - acc: 0.9418
Epoch 15/50
 - 227s - loss: 0.1612 - acc: 0.9444
Epoch 16/50
 - 7920s - loss: 0.1546 - acc: 0.9468
Epoch 17/50
 - 254s - loss: 0.1484 - acc: 0.9491
Epoch 18/50

'''




'''
Epoch 1/50 run
2018-07-18 12:05:57.464381: I tensorflow/core/platform/cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX
 - 243s - loss: 0.7457 - acc: 0.7097
Epoch 2/50
 - 244s - loss: 0.4121 - acc: 0.8518
Epoch 3/50
 - 252s - loss: 0.3522 - acc: 0.8732
Epoch 4/50
 - 240s - loss: 0.3154 - acc: 0.8867
Epoch 5/50
 - 236s - loss: 0.2880 - acc: 0.8975
Epoch 6/50
 - 234s - loss: 0.2665 - acc: 0.9054
Epoch 7/50
 - 228s - loss: 0.2477 - acc: 0.9125
Epoch 8/50
 - 228s - loss: 0.2320 - acc: 0.9182
Epoch 9/50
 - 229s - loss: 0.2181 - acc: 0.9233
Epoch 10/50
 - 228s - loss: 0.2062 - acc: 0.9279
Epoch 11/50
  - 232s - loss: 0.1955 - acc: 0.9318
Epoch 12/50
 - 242s - loss: 0.1864 - acc: 0.9351
Epoch 13/50
 - 235s - loss: 0.1779 - acc: 0.9384
Epoch 14/50
 - 233s - loss: 0.1699 - acc: 0.9413
Epoch 15/50
 - 251s - loss: 0.1630 - acc: 0.9438
Epoch 16/50
 - 235s - loss: 0.1563 - acc: 0.9463
Epoch 17/50
 - 238s - loss: 0.1501 - acc: 0.9485
Epoch 18/50
 - 236s - loss: 0.1445 - acc: 0.9506
Epoch 19/50
 - 236s - loss: 0.1391 - acc: 0.9526
Epoch 20/50
 - 233s - loss: 0.1342 - acc: 0.9543
Epoch 21/50
 - 263s - loss: 0.1293 - acc: 0.9561
Epoch 22/50
 - 253s - loss: 0.1250 - acc: 0.9576
Epoch 23/50
 - 239s - loss: 0.1207 - acc: 0.9591
Epoch 24/50
 - 238s - loss: 0.1167 - acc: 0.9606
Epoch 25/50
 - 237s - loss: 0.1129 - acc: 0.9620
Epoch 26/50
 - 243s - loss: 0.1090 - acc: 0.9634
Epoch 27/50
 - 238s - loss: 0.1056 - acc: 0.9646
Epoch 28/50
 - 240s - loss: 0.1024 - acc: 0.9657
Epoch 29/50
 - 237s - loss: 0.0990 - acc: 0.9669
Epoch 30/50
 - 236s - loss: 0.0960 - acc: 0.9680
Epoch 31/50
 - 237s - loss: 0.0930 - acc: 0.9691
Epoch 32/50
 - 240s - loss: 0.0901 - acc: 0.9701
Epoch 33/50
 - 241s - loss: 0.0877 - acc: 0.9709
Epoch 34/50
 - 251s - loss: 0.0849 - acc: 0.9720
Epoch 35/50
 - 240s - loss: 0.0825 - acc: 0.9728
Epoch 36/50
 - 240s - loss: 0.0800 - acc: 0.9737
Epoch 37/50
 - 237s - loss: 0.0777 - acc: 0.9744
Epoch 38/50
 - 240s - loss: 0.0754 - acc: 0.9752
Epoch 39/50
 - 238s - loss: 0.0733 - acc: 0.9760
Epoch 40/50
 - 239s - loss: 0.0712 - acc: 0.9768
Epoch 41/50
 - 231s - loss: 0.0690 - acc: 0.9775
Epoch 42/50
 - 231s - loss: 0.0671 - acc: 0.9782
Epoch 43/50
 - 233s - loss: 0.0652 - acc: 0.9788
Epoch 44/50
 - 231s - loss: 0.0633 - acc: 0.9796
Epoch 45/50
 - 232s - loss: 0.0617 - acc: 0.9799
Epoch 46/50
 - 264s - loss: 0.0598 - acc: 0.9808
Epoch 47/50
 - 239s - loss: 0.0580 - acc: 0.9814
Epoch 48/50
 - 241s - loss: 0.0564 - acc: 0.9819
Epoch 49/50
 - 237s - loss: 0.0550 - acc: 0.9824
Epoch 50/50
 - 237s - loss: 0.0534 - acc: 0.9829

'''