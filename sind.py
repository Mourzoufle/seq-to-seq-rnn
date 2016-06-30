'''
Pre-processing module for Microsoft SIND dataset
'''

import json
import re

def pre_process(path_txt, path_img, path_out, vocab):
    '''
    Pre-processing function to produce images-sentences pair in a subset of SIND
    '''
    items_txt = []
    with open(path_txt, 'r') as file_in:
        sents = json.load(file_in)['annotations']
        for sent in sents:
            item = []
            tokens = re.split(' *', re.sub('[^a-z\[\]]', ' ', sent[0]['text']))
            for token in tokens:
                if vocab.has_key(token):
                    item.append(vocab[token])
            items_txt.append(item)

    for i in range(0, len(items_txt), 5):
        for j in range(1, 5):
            items_txt[i].extend(items_txt[i + j])
            items_txt[i + j] = []
    items_txt = items_txt[: : 5]

    items_img = []
    with open(path_img, 'r') as file_in:
        items_img = json.load(file_in)

    with open(path_out, 'w') as file_out:
        json.dump(zip(items_img, items_txt), file_out)


if __name__ == '__main__':
    vocab = {}
    with open('dict_10.txt', 'r') as file_in:   # currently use words whose frequencies are not lower than 10
        words = file_in.readlines()
        for word in words:
            idx = len(vocab) + 1
            vocab[word.split('\t')[0]] = idx

    pre_process('train.SIS.json', 'train_img.json', 'train.json', vocab)
    pre_process('val.SIS.json', 'val_img.json', 'val.json', vocab)
    pre_process('test.SIS.json', 'test_img.json', 'test.json', vocab)
