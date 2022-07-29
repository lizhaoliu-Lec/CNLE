from numpy import array
from numpy import argmax
from math import log


# greedy decoder
def greedy_decoder(data):
    # 每一行最大概率词的索引
    return [argmax(s) for s in data]


# beam search
def beam_search_decoder(data, k):
    sequences = [[list(), 1.0]]
    for row in data:
        all_candidates = list()
        for i in range(len(sequences)):
            seq, score = sequences[i]
            for j in range(len(row)):
                candidate = [seq + [j], score * -log(row[j])]
                all_candidates.append(candidate)
        # 所有候选根据分值排序
        ordered = sorted(all_candidates, key=lambda tup: tup[1])
        # 选择前k个
        sequences = ordered[:k]
    return sequences


# 定义一个句子，长度为10，词典大小为5
data = [[0.1, 0.2, 0.3, 0.4, 0.5],
        [0.5, 0.4, 0.3, 0.2, 0.1],
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.5, 0.4, 0.3, 0.2, 0.1],
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.5, 0.4, 0.3, 0.2, 0.1],
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.5, 0.4, 0.3, 0.2, 0.1],
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.5, 0.4, 0.3, 0.2, 0.1]]
data = array(data)
# 使用greedy search解码
print('greedy')
result = greedy_decoder(data)
print(result)

print('beam search')
result = beam_search_decoder(data, 5)
# print result
for seq in result:
    print(seq)
