# -*- coding: utf-8 -*-
"""
@Author  : Shaobo Zhang
@Project : Workshop
@FileName: model.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from random import random


class BaseSeq2Seq(nn.Module):
    def __init__(self, args):
        super(BaseSeq2Seq, self).__init__()
        self.encoder = Encoder(args.vocab_size, args.embed_size, args.hidden_size, args.enc_dropout)
        self.decoder = Decoder(args.vocab_size, args.embed_size, args.hidden_size, args.dec_dropout)
        self.attn = Attention(args.hidden_size)
        self.reducer = Reduce_State(args.hidden_size)
        self.fc = nn.Linear(2 * args.hidden_size, args.vocab_size)
        self.teacher_force = args.teacher_force

    def forward(self, src_vecs, src_mask, tgt_vecs):
        """
        :param src_vecs: (batch_sz, src_len)
        :param src_mask: (batch_sz, 1, src_len)
        :param tgt_vecs: (batch_sz, tgt_len)
        :return batch_loss: batch loss with shape (1)
        """
        batch_size, tgt_len = tgt_vecs.shape
        # (batch_sz, src_len, 2*hidden_sz), (batch_sz, 2*hidden_sz)
        enc_output, enc_hidden = self.encoder(src_vecs)
        # (batch_sz, 2*hidden_sz) -> (1, batch_sz, hidden_sz)
        dec_hidden = self.reducer(enc_hidden)
        # (batch_sz, 1)
        dec_input = tgt_vecs[:, [0]]
        batch_loss = []
        for t in range(1, tgt_len):
            # (batch_sz, 1, hidden_sz), (1, batch_sz, hidden_sz)
            dec_output, dec_hidden = self.decoder(dec_input, dec_hidden)
            # (batch_sz, 2*hidden_sz)
            context_vec = self.attn(enc_output, src_mask, dec_hidden)
            # (batch_sz, 2*hidden_sz) -> (1, batch_sz, hidden_sz)
            dec_hidden = self.reducer(context_vec)
            # (batch_sz, 1, 2*hidden_sz) -> (batch_sz, 1, vocab_size)
            dec_output = self.fc(torch.cat([dec_hidden.transpose(0, 1), dec_output], dim=-1))
            p_vocab = torch.softmax(dec_output.squeeze(), dim=-1)
            loss = self.cal_loss(p_vocab, tgt_vecs[:, [t]])
            batch_loss.append(loss)
            # teacher forcing
            dec_input = tgt_vecs[:, [t]] if random() < self.teacher_force else dec_output.argmax(-1)
        # (batch_sz, tgt_len-1) -> (batch_sz)
        batch_loss = torch.sum(torch.stack(batch_loss, dim=1), dim=1)
        # (batch_sz, tgt_len)
        tgt_mask = torch.ne(tgt_vecs, 0).float()
        # (batch_sz, tgt_len) -> (batch_sz)
        tgt_lens = torch.sum(tgt_mask, dim=1) - 1
        # (batch_sz) -> (1)
        batch_loss = torch.mean(batch_loss / tgt_lens)
        return batch_loss

    def predict(self, dec_input, dec_hidden, enc_output, src_mask):
        """
        :param dec_input: (batch_sz, 1)
        :param dec_hidden: (1, batch_sz, hidden_sz)
        :param enc_output: (batch_sz, src_len, 2*hidden_sz)
        :param src_mask: (batch_sz, 1, src_len)
        :return log_probs: (vocab_size)
        :return dec_hidden: (1, batch_sz, hidden_sz)
        """
        # (batch_sz, 1, hidden_sz), (1, batch_sz, hidden_sz)
        dec_output, dec_hidden = self.decoder(dec_input, dec_hidden)
        # (batch_sz, 2*hidden_sz)
        context_vec = self.attn(enc_output, src_mask, dec_hidden)
        # (batch_sz, 2*hidden_sz) -> (1, batch_sz, hidden_sz)
        dec_hidden = self.reducer(context_vec)
        # (batch_sz, 1, 2*hidden_sz) -> (batch_sz, 1, vocab_size)
        dec_output = self.fc(torch.cat([dec_hidden.transpose(0, 1), dec_output], dim=-1))
        # (batch_sz, 1, vocab_size) -> (vocab_size)
        log_probs = F.log_softmax(dec_output.squeeze(), dim=-1)
        # (vocab_size), (1, batch_sz, hidden_sz)
        return log_probs, dec_hidden

    @staticmethod
    def cal_loss(p_vocab, target):
        """
        :param p_vocab: (batch_sz, vocab_size)
        :param target: (batch_sz, 1)
        :return loss: (batch_sz)
        """
        # (batch_sz, 1)
        target_prob = torch.gather(input=p_vocab, dim=1, index=target)
        mask = torch.ne(target, 0).float()
        # (batch_sz, 1)
        loss = -torch.log(target_prob + 1e-9) * mask
        # (batch_sz)
        return loss.squeeze()


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, dropout):
        super(Encoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(embed_size, hidden_size, bidirectional=True, batch_first=True)

    def forward(self, enc_input):
        """
        :param enc_input: (batch_sz, src_len)
        :return enc_output: (batch_sz, src_len, 2*hidden_sz)
        :return enc_hidden: (batch_sz, 2*hidden_sz)
        """
        # (batch_sz, src_len) -> (batch_sz, src_len, embed_sz)
        enc_embed = self.dropout(self.embed(enc_input))
        # (batch_sz, src_len, 2*hidden_sz), (2, batch_sz, hidden_sz)
        enc_output, enc_hidden = self.rnn(enc_embed)
        # (2, batch_sz, hidden_sz) -> (batch_sz, 2*hidden_sz)
        enc_hidden = torch.cat([enc_hidden[0], enc_hidden[1]], dim=-1)
        return enc_output, enc_hidden


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, dropout):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(embed_size, hidden_size, batch_first=True)

    def forward(self, dec_input, dec_hidden):
        """
        :param dec_input: (batch_sz, 1)
        :param dec_hidden: (1, batch_sz, hidden_sz)
        :return dec_output: (batch_sz, 1, hidden_sz)
        :return dec_hidden: (1, batch_sz, hidden_sz)
        """
        # (batch_sz, 1) -> (batch_sz, 1, embed_sz)
        dec_embed = self.dropout(self.embed(dec_input))
        # (batch_sz, 1, hidden_sz), (1, batch_sz, hidden_sz)
        dec_output, dec_hidden = self.rnn(dec_embed, dec_hidden)
        return dec_output, dec_hidden


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.W = nn.Linear(3 * hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)

    def forward(self, enc_output, enc_mask, dec_hidden):
        """
        :param enc_output: (batch_sz, src_len, 2*hidden_sz)
        :param enc_mask: (batch_sz, src_len, 1)
        :param dec_hidden: (1, batch_sz, hidden_sz)
        :return context_vec: (batch_sz, 2*hidden_sz)
        """
        enc_mask = (1 - enc_mask).bool()
        src_len = enc_output.shape[1]
        # (1, batch_sz, hidden_sz) -> (batch_sz, 1, hidden_sz) -> (batch_sz, src_len, hidden_sz)
        dec_hidden = dec_hidden.transpose(0, 1).expand(-1, src_len, -1)
        # (batch_sz, src_len, 3*hidden_sz) -> (batch_sz, src_len, hidden_sz)
        energy = self.W(torch.cat([enc_output, dec_hidden], dim=-1))
        # (batch_sz, src_len, hidden_sz) -> (batch_sz, src_len, 1) -> (batch_sz, 1, src_len)
        attn_weight = self.V(torch.tanh(energy))
        attn_weight = attn_weight.masked_fill(enc_mask, float('-inf'))
        # (batch_sz, src_len, 1) -> (batch_sz, 1, src_len)
        attn_weight = F.softmax(attn_weight.transpose(1, 2), dim=-1)
        # (batch_sz, 1, src_len) * (batch_sz, src_len, 2*hidden_sz) -> (batch_sz, 1, 2*hidden_sz)
        context_vec = torch.bmm(attn_weight, enc_output)
        # (batch_sz, 1, 2*hidden_sz) -> (batch_sz, 2*hidden_sz)
        return context_vec.squeeze(1)


class Reduce_State(nn.Module):
    def __init__(self, hidden_size):
        super(Reduce_State, self).__init__()
        self.fc = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, enc_hidden):
        """
        :param enc_hidden: (batch_sz, 2*hidden_size)
        :return dec_hidden: (1, batch_sz, hidden_sz)
        """
        dec_hidden = self.fc(enc_hidden).unsqueeze(0)
        return dec_hidden
