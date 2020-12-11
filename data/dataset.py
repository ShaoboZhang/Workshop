# -*- coding: utf-8 -*-
"""
@Author  : Shaobo Zhang
@Project : Workshop
@FileName: dataset.py
"""

import torch
import torch.utils.data as Data
from data.data_utils import Vocab, source2id
from model.config import pointer


class PairDataset(Data.Dataset):
    def __init__(self, file_path, args, vocab=None):
        super(PairDataset, self).__init__()
        # Reading file
        print(f'读取数据...', end=' ')
        pairs = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                pair = line.split('<sep>')
                src = pair[0].strip().split()[:args.max_src_len]
                tgt = pair[1].strip().split()[:args.max_tgt_len]
                pairs.append((src, tgt))
        print(f'共读取数据{len(pairs)}对')

        # Building vocabulary
        self.vocab = vocab if vocab else Vocab(pairs, args.vocab_size)
        self.pairs = pairs
        self.max_src_len, self.max_tgt_len, self.pointer = args.max_src_len, args.max_tgt_len, args.pointer

    def __getitem__(self, item):
        return source2id(self.vocab, self.pairs[item], self.max_src_len, self.max_tgt_len, self.pointer)

    def __len__(self):
        return len(self.pairs)

    def get_vocab(self):
        return self.vocab


def collate_fn(batch):
    src_vecs, src_masks, tgt_vecs = [], [], []
    oov_dicts = [] if pointer else None
    for data in batch:
        src_vecs.append(data[0])
        src_masks.append(data[1])
        tgt_vecs.append(data[2])
        if pointer:
            oov_dicts.append(data[-1])
    src_vecs, src_masks, tgt_vecs = map(torch.tensor, (src_vecs, src_masks, tgt_vecs))
    if pointer:
        return src_vecs, src_masks, tgt_vecs, oov_dicts
    else:
        return src_vecs, src_masks, tgt_vecs
