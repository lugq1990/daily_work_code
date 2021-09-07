# -*- coding:utf-8 -*-
import numpy as np
import re

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


# Load data from local files
def load_data(pos_path, neg_path):
    pos = list(open(pos_path, 'r', encoding='utf-8').readline())
    pos = [s.strip() for s in pos]
    neg = list(open(neg_path, 'r', encoding='utf-8').readline())
    neg = [s.strip() for s in neg]

    text_data = pos + neg
    text_data = [clean_str(s) for s in text_data]

    pos_label = [[0, 1] for _ in pos]
    neg_label = [[1, 0] for _ in neg]
    label = np.concatenate([pos_label, neg_label], axis=0)

    return [text_data, label]

# Here is a function to produce batch data
def batch_iter(data, batch_size, num_epohcs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    batch_epoch = int((data_size -1)/batch_size) + 1
    for e in range(num_epohcs):
        if shuffle:
            shuffle_index = np.random.permutation(np.arange(data_size))
            shuffle_data = data[shuffle_index]
        else: shuffle_data = data
        for b in range(batch_epoch):
            start = b * batch_size
            end = max((b+1)*batch_size, data_size)
            yield shuffle_data[start: end]

