import numpy as np

#转移概率，单纯用了等概率
trans_prob = {
    'be':0.5,
    'bm':0.5,
    'eb':0.5,
    'es':0.5,
    'me':0.5,
    'mm':0.5,
    'sb':0.5,
    'ss':0.5
}
trans_prob = {i:np.log(trans_prob[i]) for i in trans_prob.keys()}

def viterbi(nodes,trans_prob = trans_prob):
    # key即是一条路径,每个char存对应node的label
    # 这里的label相当于hidden state
    paths = {'b':nodes[0]['b'], 's':nodes[0]['s']}
    # paths在任意时刻,最多只有4个key-value对
    # 以s结尾的最优路径的概率
    # 以b结尾的最优路径的概率
    # 以e结尾的最优路径的概率
    # 以m结尾的最优路径的概率
    for node_idx in range(1,len(nodes)):
        paths_ = paths.copy()
        paths = {}
        for label in nodes[node_idx].keys():
            #内循环 求以给定的label(hidden state)作为最后一个label的最优路径
            nows = {}
            for path in paths_.keys():
                pre_node_label = path[-1]
                if pre_node_label + label in trans_prob.keys():
                    # 把概率相乘等价地转化为相加
                    nows[path+label]= paths_[path]+trans_prob[path[-1]+label]+nodes[node_idx][label]
            k = np.argmax(list(nows.values()))
            paths[list(nows.keys())[k]] = list(nows.values())[k]
    return list(paths.keys())[np.argmax(list(paths.values()))]


if __name__ == '__main__':
    test_nodes = [{'s': 0.2, 'b': 0.2, 'm': 0.3, 'e': 0.3}, {'s': 0.1, 'b': 0.2, 'm': 0.3, 'e': 0.4}]
    print((viterbi(test_nodes)))
    # be