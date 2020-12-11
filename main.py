# -*- coding: utf-8 -*-
"""
@Author  : Shaobo Zhang
@Project : Workshop
@FileName: main.py
"""

from data.dataset import PairDataset, collate_fn
from model import config
from model.model import BaseSeq2Seq
from model.pgn import PGN
from model.utils import train_eval
from model.predict import BasePredict, PGNPredict

import torch.utils.data as Data
from time import time


def timeit(func):
    def wrapper(*args, **kwargs):
        t0 = time()
        func(*args, **kwargs)
        t1 = time()
        print(f"Processing time: {(t1 - t0) / 60:.2f}min")

    return wrapper


@timeit
def model_train():
    # Generate train dataset
    train_dataset = PairDataset(config.train_data_path, config)
    train_data = Data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    vocab = train_dataset.get_vocab()
    # Generate validation dataset
    valid_dataset = PairDataset(config.valid_data_path, config, vocab)
    valid_data = Data.DataLoader(valid_dataset, batch_size=config.batch_size, collate_fn=collate_fn)
    model = PGN(config, vocab) if config.pointer else BaseSeq2Seq(config)
    train_eval(model, train_data, valid_data, config)


def summary():
    # Get vocab dict
    train_dataset = PairDataset(config.train_data_path, config)
    vocab = train_dataset.get_vocab()
    # Generate test dataset
    test_dataset = PairDataset(config.test_data_path, config, vocab)
    test_data = Data.DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn)
    model = PGN(config, vocab) if config.pointer else BaseSeq2Seq(config)
    predictor = PGNPredict(model, vocab, config) if config.pointer else BasePredict(model, vocab, config)
    f1_r1, f1_r2, f1_rl = predictor.predict(test_data, beam_search=True)
    print("Rouge-1 f1_score: ", f1_r1)
    print("Rouge-2 f1_score: ", f1_r2)
    print("Rouge-L f1_score: ", f1_rl)
    return f1_r1, f1_r2, f1_rl


if __name__ == '__main__':
    # model_train()
    summary()
    pass
