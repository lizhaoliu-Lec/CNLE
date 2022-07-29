from sklearn import metrics
from typing import List, Tuple, Dict, Set, Union
import matplotlib.pyplot as plt
import numpy as np
import json
import math

""" Programm for computing metrics for the given result"""

dataset_dir = "runs//AAPD//dualTransV4Ori//2019-12-06...20.38.48//iteration_39000/"
dataset_name = 'AAPD'

'''Can add detailed information for this result'''
detail_info = ""


def fopen():
    """Open file and extract List[List[Str]].
    """
    # file_total_label = dataset_name + "/label_train"
    file_test = dataset_dir + "/" + dataset_name + ".gold" + detail_info + ".txt"
    file_predict = dataset_dir + "/" + dataset_name + detail_info + ".txt"

    # with open(file_total_label, "r", encoding="utf-8") as reader:
    #     total_labels = reader.readlines()
    with open(file_test, "r", encoding="utf-8") as reader:
        references = reader.readlines()
    with open(file_predict, "r", encoding="utf-8") as reader:
        hypotheses = reader.readlines()

    refers = [reference.replace('1', 'A').replace('2', 'B').replace('3', 'C').replace('4', 'D').replace('5', 'E')
                  .replace('6', 'F').replace('7', 'G').replace('8', 'H').replace('9', 'I').replace('.', '')
                  .replace('"', '').replace('-', '').lower().split()
              for reference in references]
    hypos = [hypothese.split() for hypothese in hypotheses]
    # hypos = [list(set(hypothese.split())) for hypothese in hypotheses]

    return refers, hypos


def create_idx(labels):
    """Create label to index dict."""
    idx = 0
    l2i_dict = {}
    for label in labels:
        # label = label.split()
        for la in label:
            if la not in l2i_dict.keys():
                l2i_dict[la] = idx
                idx += 1
    print("")
    return l2i_dict


def tag2Idx(tags, l2i_dict):
    """ Return the indexes of the given tags"""
    return [l2i_dict[tag] for tag in tags]


def compute_hamming_loss(references, hypotheses, tgt_dictionary):
    """ Given decoding results and reference sentences, compute corpus-level BLEU score.
    @param references: (List[List[str]]), a list of gold-standard reference target sentences (or labels)
    @param hypotheses: (List[predict]), a list of hypotheses, one for each reference
    @param tgt_dictionary: (Dict[str, int]), a dictionary of tgt sentences (or labels)
    @returns hamming_loss: hamming loss
    """

    def sentence_ids_to_multi_ones_hot_vector(y, dictionary):
        total_length = len(dictionary)
        ones_hot = np.zeros(total_length, dtype=np.int)
        hot_indices = tag2Idx(y, dictionary)
        ones_hot[hot_indices] = 1
        # ignore the following words '<pad>' '<s>' '</s>' '<unk>'
        return ones_hot

    def sentences_ids_to_multi_ones_hot_vectors(ys, dictionary):
        return np.array([sentence_ids_to_multi_ones_hot_vector(y, dictionary) for y in ys],
                        dtype=np.int)

    references_ones_hot_vectors = sentences_ids_to_multi_ones_hot_vectors(references, tgt_dictionary)
    hypotheses_ones_hot_vectors = sentences_ids_to_multi_ones_hot_vectors(hypotheses,
                                                                          tgt_dictionary)
    hamming_loss = metrics.hamming_loss(references_ones_hot_vectors, hypotheses_ones_hot_vectors)
    macro_f1 = metrics.f1_score(references_ones_hot_vectors, hypotheses_ones_hot_vectors, average='macro')
    macro_precision = metrics.precision_score(references_ones_hot_vectors, hypotheses_ones_hot_vectors, average='macro')
    macro_recall = metrics.recall_score(references_ones_hot_vectors, hypotheses_ones_hot_vectors, average='macro')

    micro_f1 = metrics.f1_score(references_ones_hot_vectors, hypotheses_ones_hot_vectors, average='micro')
    micro_precision = metrics.precision_score(references_ones_hot_vectors, hypotheses_ones_hot_vectors, average='micro')
    micro_recall = metrics.recall_score(references_ones_hot_vectors, hypotheses_ones_hot_vectors, average='micro')

    results = dict(hamming_loss=round(hamming_loss, 4),
                   macro_f1=round(macro_f1, 3),
                   macro_precision=round(macro_precision, 3),
                   macro_recall=round(macro_recall, 3),
                   micro_f1=round(micro_f1, 3),
                   micro_precision=round(micro_precision, 3),
                   micro_recall=round(micro_recall, 3), )

    return results


if __name__ == "__main__":
    references, hypotheses = fopen()
    l2i_dict = create_idx(references)
    # print("-----l2i_dict-----\n", l2i_dict)
    results = compute_hamming_loss(references, hypotheses, l2i_dict)
    print("-----Result-----\n", results)
