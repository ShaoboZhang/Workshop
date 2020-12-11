# -*- coding: utf-8 -*-
"""
@Author  : Shaobo Zhang
@Project : Workshop
@FileName: data_utils.py
"""

from collections import Counter
import numpy as np
import torch


class Vocab:
    PAD = 0
    UNK = 1
    BOS = 2
    EOS = 3

    def __init__(self, sentences, vocab_size):
        """
        Build vocabulary dictionary
        :param sentences: sentences list {(src, tgt)} read from file
        :param vocab_size: max vocab size
        """
        self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
        self.idx2word = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        # building vocabulary
        print('构建词典...', end=' ')
        word_count = Counter()
        for src_sent, tgt_sent in sentences:
            word_count.update(src_sent)
            # word_count.update(tgt_sent)
        for idx, (word, _) in enumerate(word_count.most_common(vocab_size - 4)):
            self.word2idx[word] = idx + 4
            self.idx2word.append(word)
        print(f'共{len(self.idx2word)}个单词')

    def __getitem__(self, item):
        if type(item) == int:
            return self.idx2word[item]
        else:
            return self.word2idx.get(item, self.UNK)

    def __len__(self):
        return len(self.idx2word)


class Beam(object):
    def __init__(self, seq_idx, log_probs, dec_hidden):
        self.seq_idx = seq_idx
        self.log_probs = log_probs
        self.dec_hidden = dec_hidden

    def extend(self, seq_idx, log_probs, dec_hidden):
        seq_idx = torch.cat([self.seq_idx, seq_idx])
        seq_len = len(self.seq_idx)
        log_probs = (log_probs + self.log_probs * seq_len) / (1 + seq_len)
        return Beam(seq_idx, log_probs, dec_hidden)

    def __len__(self):
        return len(self.seq_idx)

    def __lt__(self, other):
        return self.log_probs < other.log_probs

    def __le__(self, other):
        return self.log_probs <= other.log_probs


def source2id(vocab: Vocab, sentence, max_src_len, max_tgt_len, pointer=False):
    """
    Transfer sentences to index
    :param vocab: the built vocab dictionary
    :param sentence: sentences (src, tgt) read from file
    :param max_src_len: max length of source sentence
    :param max_tgt_len: max length of target sentence
    :param pointer: whether use pointer network
    :return src_vecs: padded source vectors
    :return src_mask: source vectors masks
    :return tgt_vecs: padded target vectors
    """
    src_sent, tgt_sent = sentence
    src_mask = np.zeros((max_src_len, 1))
    oov_dict = {} if pointer else None
    if pointer:
        src_vec, tgt_vec = [], []
        for word in src_sent:
            if word not in vocab.idx2word:
                if word not in oov_dict:
                    oov_dict[word] = len(vocab) + len(oov_dict)
                src_vec.append(oov_dict[word])
            else:
                src_vec.append(vocab[word])
        for word in tgt_sent:
            if word in oov_dict:
                tgt_vec.append(oov_dict[word])
            else:
                tgt_vec.append(vocab[word])
    else:
        src_vec = [vocab[word] for word in src_sent]
        tgt_vec = [vocab[word] for word in tgt_sent]
    src_vec = [vocab.BOS] + src_vec[:max_src_len - 2] + [vocab.EOS]
    tgt_vec = [vocab.BOS] + tgt_vec[:max_tgt_len - 2] + [vocab.EOS]
    # padding
    src_vec += [0] * (max_src_len - len(src_vec))
    tgt_vec += [0] * (max_tgt_len - len(tgt_vec))
    src_mask[:len(src_vec), :] = 1
    if pointer:
        return src_vec, src_mask, tgt_vec, oov_dict
    else:
        return src_vec, src_mask, tgt_vec



def id2output(tokens, vocab: Vocab, oov_dict:dict=None):
    """
    Transfer id to sentence
    """
    output = []
    oov_keys = list(oov_dict.keys()) if oov_dict else None
    for idx in tokens:
        if idx == vocab.PAD or idx == vocab.EOS:
            break
        if idx != vocab.BOS:
            if oov_dict and idx >= len(vocab):
                output.append(oov_keys[idx-len(vocab)])
            else:
                output.append(vocab[idx])
    return " ".join(output)


def replace_oov(tokens, vocab: Vocab, device):
    """
    Replace oov tokens by UNK
    """
    oov_token = torch.full(tokens.shape, vocab.UNK).long().to(device)
    output_tokens = torch.where(tokens >= len(vocab), oov_token, tokens)
    # tokens[tokens >= len(vocab)] = vocab.UNK  # inplace operation error
    return output_tokens
