'''
Pre-processing module for Microsoft SIND dataset
'''

import json
import re


def main(
    threshold=10,   # Currently filter words with frequency lower than 10 - Treat them as UNK in the vocabulary
    path_txt_train='train.SIS.json',
    path_txt_val='val.SIS.json',
    path_txt_test='test.SIS.json',
    path_img_train='train_img.json',
    path_img_val='val_img.json',
    path_img_test='test_img.json',
    path_vocab='vocab.json',
    path_out_train='train.json',
    path_out_val='val.json',
    path_out_test='test.json',
    ):
    '''
    Main function
    '''
    paths_txt = [path_txt_train, path_txt_val, path_txt_test]
    paths_img = [path_img_train, path_img_val, path_img_test]
    paths_out = [path_out_train, path_out_val, path_out_test]

    subs = [[r'[,.!?]', ' '], ['\'ll ', ' will '], ['\'d ', ' would '], ['\'n\'', ' and '], ['e\'s ', 'e is '], [' i\'m ', ' i am '], [' they\'re ', ' they are '], [r'[^a-z\[\]]', ' ']]
    vocab = {}
    sets_txt = []

    for path in paths_txt:
        sets_txt.append([])
        with open(path, 'r') as file_in:
            sents = json.load(file_in)['annotations']
            for sent in sents:
                sent = sent[0]['text']
                for sub in subs:
                    sent = re.sub(sub[0], sub[1], sent)
                tokens = sent.split(' ')
                sent = []
                for token in tokens:
                    if not token:
                        continue
                    sent.append(token)
                    if token in vocab:
                        vocab[token] += 1
                    else:
                        vocab[token] = 1
                sets_txt[-1].append(sent)

    idx = 1
    for key in vocab.keys():
        if vocab[key] < threshold:
            del vocab[key]
        else:
            vocab[key] = idx
            idx += 1
    vocab['EOS'] = 0
    vocab['UNK'] = idx
    with open(path_vocab, 'w') as file_out:
        json.dump(vocab, file_out)

    for i, set_txt in enumerate(sets_txt):
        for sent in set_txt:
            for idx, token in enumerate(sent):
                sent[idx] = vocab.get(token, idx)
            sent.append(0)
        for j in range(0, len(set_txt), 5):
            set_txt[j] = [set_txt[j + k] for k in range(5)]
        set_txt = set_txt[: : 5]

        with open(paths_img[i], 'r') as file_in:
            set_img = json.load(file_in)
        with open(paths_out[i], 'w') as file_out:
            json.dump(zip(set_img, set_txt), file_out)


if __name__ == '__main__':
    main()
