import re
import os
import json
from tqdm import tqdm


pure_texts = []
pure_tags = []
stops = u'，。！？；、：,\.!\?;:\n'
for txt in tqdm(open('../data/msr_train.txt',encoding='utf8')):
    # txt 相当于一行
    print(txt)
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

    break

print(pure_texts)
print('ch')
for item in pure_texts[0]:
    print(item,end = '    ')

print()
print(pure_tags)