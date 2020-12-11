# -*- coding: utf-8 -*-
"""
@Author  : Shaobo Zhang
@Project : Workshop
@FileName: config.py
"""

import torch

# data
train_data_path = './files/train.txt'
valid_data_path = './files/dev.txt'
test_data_path = './files/test.txt'

max_src_len: int = 250  # exclusive of BOS and EOS
max_tgt_len: int = 80  # exclusive of BOS and EOS

# model
vocab_size = 50000
embed_size: int = 300
hidden_size: int = 128
enc_dropout = 0.3
dec_dropout = 0.4
teacher_force = 0.5

# pointer network
pointer = False
lamb = 1

# train/test
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 64
num_epochs = 10
learning_rate = 2e-4

# model save
# model_path = './save/basic_seq2seq.pt'
model_path = f'./save/pgn_{vocab_size//1000}k.pt' if pointer else f'./save/seq2seq_{vocab_size//1000}k.pt'

# predict summary
beam_width = 5
