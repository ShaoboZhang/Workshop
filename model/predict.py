# -*- coding: utf-8 -*-
"""
@Author  : Shaobo Zhang
@Project : Workshop
@FileName: predict.py
@Discribe: 
"""

import torch
import numpy as np
from time import sleep
from tqdm import tqdm
from rouge import Rouge
from data.data_utils import Beam, id2output, replace_oov


class BasePredict:
    def __init__(self, model, vocab, args):
        model_dict = torch.load(args.model_path)
        model.load_state_dict(model_dict)
        self.model = model.to(args.device)
        self.vocab = vocab
        self.device = args.device
        self.max_len = args.max_tgt_len
        self.beam_width = args.beam_width

    def predict(self, test_data, beam_search=True):
        """
        :param test_data: test data loader
        :param beam_search: whether use beam search, otherwise use greedy search
        :return:
        """
        rouge = Rouge()
        f1_r1, f1_r2, f1_rl = [], [], []
        print("生成摘要...")
        sleep(0.5)
        self.model.eval()
        count = 0
        with torch.no_grad():
            # batch_sz = 1
            for src_vec, src_mask, tgt_vec in tqdm(test_data):
                # (batch_sz, src_len)
                src_vec = src_vec.to(self.device)
                # (batch_sz, 1, src_len)
                src_mask = src_mask.to(self.device)
                # (batch_sz, src_len, 2*hidden_sz), (2, batch_sz, hidden_sz)
                enc_output, enc_hidden = self.model.encoder(src_vec)
                # (2, batch_sz, hidden_sz) -> (1, batch_sz, hidden_sz)
                dec_hidden = self.model.reducer(enc_hidden)
                # (batch_sz, )
                dec_input = tgt_vec[:, 0].to(self.device)
                if beam_search:
                    summary = self.beam_search(dec_input, dec_hidden, enc_output, src_mask, self.beam_width)
                else:
                    summary = self.beam_search(dec_input, dec_hidden, enc_output, src_mask, 1)
                sentence = id2output(summary, self.vocab)
                ref = id2output(tgt_vec.squeeze().cpu().tolist(), self.vocab)
                score = rouge.get_scores(sentence, ref)[0]
                f1_r1.append(score['rouge-1']['f'])
                f1_r2.append(score['rouge-2']['f'])
                f1_rl.append(score['rouge-l']['f'])
                if (count + 1) % 100 == 0:
                    print('\r', ref)
                    print(sentence, '\n')
                count += 1
        return map(np.mean, (f1_r1, f1_r2, f1_rl))

    def beam_search(self, dec_input, dec_hidden, enc_output, src_mask, beam_width=5):
        """
        If beam_width = 1, it's Greedy Search.
        If beam_width > 1, it's Beam Search.
        :param beam_width: number of beams
        :return: summary sentence
        """
        beams = [Beam(dec_input, 0, dec_hidden)]
        for i in range(self.max_len):
            new_beams = []
            eos_num = 0
            for beam in beams:
                # Continue if one beam already ends
                if beam.seq_idx[-1] == self.vocab.EOS or len(beam.seq_idx) >= 50:
                    eos_num += 1
                    new_beams.append(beam)
                    if eos_num == beam_width:
                        return beams[0].seq_idx.cpu().tolist()
                    continue
                # Predict next word
                dec_input = beam.seq_idx[-1].reshape(1, 1)  # (batch_sz, ) -> (batch_sz, 1)
                # (vocab_size), (1, batch_sz, hidden_sz)
                log_probs, dec_hidden = self.model.predict(dec_input, beam.dec_hidden, enc_output, src_mask)
                topk_probs, topk_idx = torch.topk(log_probs, k=beam_width)
                # Each beam generates top k new beams
                for prob, idx in zip(topk_probs, topk_idx):
                    idx = idx.reshape(1)
                    new_beams += [beam.extend(idx, prob.cpu().item(), dec_hidden)]
            # In new_beams, there're k beams at first iteration and k*k beams otherwise.
            new_beams = sorted(new_beams, reverse=True)
            beams = new_beams[:beam_width]
        return beams[0].seq_idx.cpu().tolist()


class PGNPredict:
    def __init__(self, model, vocab, args):
        model_dict = torch.load(args.model_path)
        model.load_state_dict(model_dict)
        self.model = model.to(args.device)
        self.vocab = vocab
        self.device = args.device
        self.max_len = args.max_tgt_len
        self.beam_width = args.beam_width

    def predict(self, test_data, beam_search=True):
        """
        :param test_data: test data loader
        :param beam_search: whether use beam search, otherwise use greedy search
        :return:
        """
        rouge = Rouge()
        f1_r1, f1_r2, f1_rl = [], [], []
        print("生成摘要...")
        sleep(0.5)
        self.model.eval()
        count = 0
        with torch.no_grad():
            # batch_sz = 1
            for batch_data in tqdm(test_data):
                # (batch_sz, src_len), (batch_sz, src_len, 1), (batch_sz, tgt_len)
                src_vec, src_mask, tgt_vec = [data.to(self.device) for data in batch_data[:-1]]
                oov_dict = batch_data[-1]
                # (batch_sz, src_len, 2*hidden_sz), (batch_sz, 2*hidden_sz)
                enc_output, enc_hidden = self.model.encoder(replace_oov(src_vec, self.vocab, self.device))
                # (2, batch_sz, hidden_sz) -> (1, batch_sz, hidden_sz)
                context_vec = self.model.reducer(enc_hidden)
                # (batch_sz, )
                dec_input = tgt_vec[:, 0].to(self.device)
                # (batch_sz, src_len, 1)
                coverage_vec = torch.zeros_like(src_vec, dtype=torch.float).unsqueeze(2)
                if beam_search:
                    summary = self.beam_search(dec_input, context_vec, coverage_vec, enc_output, src_mask, src_vec,
                                               oov_dict, self.beam_width)
                else:
                    summary = self.beam_search(dec_input, context_vec, coverage_vec, enc_output, src_mask, src_vec,
                                               oov_dict, 1)
                sentence = id2output(summary, self.vocab, oov_dict[0])
                ref = id2output(tgt_vec.squeeze().cpu().tolist(), self.vocab, oov_dict[0])
                score = rouge.get_scores(sentence, ref)[0]
                f1_r1.append(score['rouge-1']['f'])
                f1_r2.append(score['rouge-2']['f'])
                f1_rl.append(score['rouge-l']['f'])
                if (count + 1) % 100 == 0:
                    print('\r', ref)
                    print(sentence, '\n')
                count += 1
        return map(np.mean, (f1_r1, f1_r2, f1_rl))

    def beam_search(self, dec_input, context_vec, coverage_vec, enc_output, src_mask, src_vec, oov_dict, beam_width=5):
        """
        If beam_width = 1, it's Greedy Search.
        If beam_width > 1, it's Beam Search.
        :param beam_width: number of beams
        :return: summary sentence
        """
        beams = [Beam(dec_input, 0, context_vec)]
        for i in range(self.max_len):
            new_beams = []
            eos_num = 0
            for beam in beams:
                # Continue if one beam already ends
                if beam.seq_idx[-1] == self.vocab.EOS or len(beam.seq_idx) >= 50:
                    eos_num += 1
                    new_beams.append(beam)
                    if eos_num == beam_width:
                        return beams[0].seq_idx.cpu().tolist()
                    continue
                # Predict next word
                dec_input = beam.seq_idx[-1].reshape(1, 1)  # (batch_sz, ) -> (batch_sz, 1)
                # (vocab_size), (1, batch_sz, hidden_sz)
                log_probs, context_vec = self.model.predict(dec_input, beam.dec_hidden, coverage_vec, enc_output,
                                                            src_mask, src_vec, oov_dict)
                topk_probs, topk_idx = torch.topk(log_probs, k=beam_width)
                # Each beam generates top k new beams
                for prob, idx in zip(topk_probs, topk_idx):
                    idx = idx.reshape(1)
                    new_beams += [beam.extend(idx, prob.cpu().item(), context_vec)]
            # In new_beams, there're k beams at first iteration and k*k beams otherwise.
            new_beams = sorted(new_beams, reverse=True)
            beams = new_beams[:beam_width]
        return beams[0].seq_idx.cpu().tolist()
