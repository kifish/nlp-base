from collections import Counter
from math import log

hmm_model = {i:Counter() for i in 'sbme'}



# dict.txt
# 词 频率或频数
'''
中国 0.01
中科院 0.001

'''
with open('dict.txt',encoding='utf8') as f:
    for line in f:
        items = line.split(' ')
        if len(items[0]) == 1: #单字词
            hmm_model['s'][items[0]] += int(items[1])
        else: #多字词
            hmm_model['b'][items[0][0]] += int(items[1])
            hmm_model['e'][items[0][-1]] += int(items[1])
            for m in items[0][1:-1]:
                hmm_model['m'][m] += int(items[1])

log_total = {i:log(sum(hmm_model[i].values())) for i in 'sbme'}

trans = {
    'ss':0.3,
    'sb':0.7,
    'bm':0.3,
    'be':0.7,
    'mm':0.3,
    'me':0.7,
    'es':0.3,
    'eb':0.7
 }

trans = {i:log(j) for i,j in trans.items()}


from viterbi import viterbi

def hmm_cut(s):
    #log 把除法转化为减法
    # j为字标记为i的所有字的list
    #j[t]表示标记为i中 t字出现的次数
    #log_total[i] 表示 标记为i的 所有字的出现总个数并取log
    # 分母为标记i出现的总次数
    # 分子为该字t的隐状态是标记i的次数
    # log相减即相除
    nodes = [{i:log(j[t]+1)-log_total[i] for i,j in hmm_model.iteritems()} for t in s]
    # 获得每个字对应的标记
    tags = viterbi(nodes)
    words = [s[0]]
    for i in range(1, len(s)):
        if tags[i] in ['b', 's']:
            words.append(s[i])
        else:
            words[-1] += s[i]
    return words
