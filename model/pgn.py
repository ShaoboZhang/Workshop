# -*- coding: utf-8 -*-
"""
@Author  : Shaobo Zhang
@Project : Workshop
@FileName: pgn.py
@Discribe: 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from random import random
from data.data_utils import replace_oov


class PGN(nn.Module):
    def __init__(self, args, vocab):
        super(PGN, self).__init__()
        # Network layers
        self.encoder = Encoder(args.vocab_size, args.embed_size, args.hidden_size, args.enc_dropout)
        self.decoder = Decoder(args.vocab_size, args.embed_size, args.hidden_size, args.dec_dropout)
        self.attention = Attention(args.hidden_size)
        self.reducer = Reduce_State(args.hidden_size)
        # Network linear matrix
        self.w_gen = nn.Linear(2 * args.hidden_size + args.embed_size, 1)
        self.fc = nn.Linear(2 * args.hidden_size, args.vocab_size)
        # Network parameters
        self.vocab = vocab
        self.device = args.device
        self.lamb = args.lamb
        self.teacher_force = args.teacher_force

    def forward(self, src_vecs, src_mask, tgt_vecs, oov_dicts):
        """
        :param src_vecs: (batch_sz, src_len)
        :param src_mask: (batch_sz, 1, src_len)
        :param tgt_vecs: (batch_sz, tgt_len)
        :param oov_dicts: oov dictionaries
        :return:
        """
        batch_size, tgt_len = tgt_vecs.shape
        # (batch_sz, src_len, 2*hidden_sz), (batch_sz, 2*hidden_sz)
        enc_output, enc_hidden = self.encoder(replace_oov(src_vecs, self.vocab, self.device))
        # (batch_sz, 2*hidden_sz) -> (1, batch_sz, hidden_sz)
        context_vec = self.reducer(enc_hidden)
        # (batch_sz, 1)
        dec_input = tgt_vecs[:, [0]]
        # (batch_sz, src_len, 1)
        coverage_vec = torch.zeros_like(src_vecs, dtype=torch.float).unsqueeze(2)
        batch_loss = []
        for t in range(1, tgt_len):
            dec_input = replace_oov(dec_input, self.vocab, self.device)
            # (batch_sz, 1, hidden_sz), (1, batch_sz, hidden_sz), (1, batch_sz, embed_sz)
            dec_output, dec_hidden, dec_embed = self.decoder(dec_input, context_vec)
            # (batch_sz, 2*hidden_sz), (batch_sz, src_len), (batch_sz, src_len, 1)
            context_vec, attn_weight, coverage_vec = self.attention(enc_output, src_mask, dec_hidden, coverage_vec)
            # (batch_sz, 2*hidden_sz) -> (1, batch_sz, hidden_sz)
            context_vec = self.reducer(context_vec)
            # (batch_sz, 1, 2*hidden_sz) -> (batch_sz, 1, vocab_size)
            dec_output = self.fc(torch.cat([context_vec.transpose(0, 1), dec_output], dim=-1))
            # (batch_sz, 1, vocab_size) -> (batch_sz, vocab_size)
            p_vocab = F.softmax(dec_output, dim=-1).squeeze(1)
            ## get p_gen
            # (1, batch_sz, 2*hidden_sz+embed_sz) -> (1, batch_sz, 1) -> (batch_sz, 1)
            score = self.w_gen(torch.cat([context_vec, dec_hidden, dec_embed], dim=-1)).squeeze(0)
            # (batch_sz, 1)
            p_gen = torch.sigmoid(score)
            ## get extended p_vocab
            # (batch_sz, extend_vocab_size)
            p_vocab = self.extend_prob(p_vocab, attn_weight, p_gen, src_vecs, oov_dicts)
            ## calculate loss
            # (batch_sz, 1)
            target = tgt_vecs[:, [t]]
            # (batch_sz)
            loss = self.cal_loss(p_vocab, target, attn_weight, coverage_vec.squeeze(2))
            batch_loss.append(loss)
            ## teacher forcing
            dec_input = target if random() < self.teacher_force else p_vocab.argmax(dim=1, keepdim=True)
        # (batch_sz, tgt_len-1) -> (batch_sz)
        batch_loss = torch.sum(torch.stack(batch_loss, dim=1), dim=1)
        # (batch_sz, tgt_len)
        tgt_mask = torch.ne(tgt_vecs, 0).float()
        # (batch_sz, tgt_len) -> (batch_sz)
        tgt_lens = torch.sum(tgt_mask, dim=1) - 1
        # (batch_sz) -> (1)
        batch_loss = torch.mean(batch_loss / tgt_lens)
        return batch_loss

    def predict(self, dec_input, context_vec, coverage_vec, enc_output, src_mask, src_vec, oov_dict):
        """
        :param dec_input: (batch_sz, 1)
        :param context_vec: (1, batch_sz, hidden_sz)
        :param coverage_vec: (batch_sz, src_len, 1)
        :param enc_output: (batch_sz, src_len, 2*hidden_sz)
        :param src_mask: (batch_sz, 1, src_len)
        :param src_vec: (batch_sz, src_len)
        :param oov_dict: oov dictionary
        :return p_vocab: (extend_vocab_size)
        :return context_vec: (1, batch_sz, hidden_sz)
        """
        dec_input = replace_oov(dec_input, self.vocab, self.device)
        # (batch_sz, 1, hidden_sz), (1, batch_sz, hidden_sz), (1, batch_sz, embed_sz)
        dec_output, dec_hidden, dec_embed = self.decoder(dec_input, context_vec)
        # (batch_sz, 2*hidden_sz), (batch_sz, src_len), (batch_sz, src_len, 1)
        context_vec, attn_weight, coverage_vec = self.attention(enc_output, src_mask, dec_hidden, coverage_vec)
        # (batch_sz, 2*hidden_sz) -> (1, batch_sz, hidden_sz)
        context_vec = self.reducer(context_vec)
        # (batch_sz, 1, 2*hidden_sz) -> (batch_sz, 1, vocab_size)
        dec_output = self.fc(torch.cat([context_vec.transpose(0, 1), dec_output], dim=-1))
        # (batch_sz, 1, vocab_size) -> (batch_sz, vocab_size)
        p_vocab = F.softmax(dec_output, dim=-1).squeeze(1)
        ## get p_gen
        # (1, batch_sz, 2*hidden_sz+embed_sz) -> (1, batch_sz, 1) -> (batch_sz, 1)
        score = self.w_gen(torch.cat([context_vec, dec_hidden, dec_embed], dim=-1)).squeeze(0)
        # (batch_sz, 1)
        p_gen = torch.sigmoid(score)
        ## get extended p_vocab
        # (batch_sz, extend_vocab_size)
        p_vocab = self.extend_prob(p_vocab, attn_weight, p_gen, src_vec, oov_dict)
        # (batch_sz, src_len)
        scatter_src = -torch.min(torch.abs(coverage_vec).squeeze(2), torch.abs(attn_weight)) * self.lamb * 0.5
        p_vocab = p_vocab.scatter_add(dim=1, index=src_vec, src=scatter_src)
        return p_vocab.squeeze(), context_vec

    def extend_prob(self, p_vocab, attn_weight, p_gen, src_vecs, oov_dicts):
        """
        :param p_vocab: (batch_sz, vocab_size)
        :param attn_weight: (batch_sz, src_len)
        :param p_gen: (batch_sz, 1)
        :param src_vecs: (batch_sz, src_len)
        :param oov_dicts: oov dictionaries
        :return p_vocab: (batch_sz, extend_vocab_size)
        """
        batch_size = len(p_gen)
        p_gen = torch.clamp(p_gen, min=0.001, max=0.999)
        extend_len = len(max(oov_dicts, key=len))
        p_extend = torch.zeros((batch_size, extend_len)).float().to(self.device)
        p_vocab = p_gen * p_vocab
        # (batch_sz, extend_vocab_size)
        p_vocab = torch.cat([p_vocab, p_extend], dim=1)
        # (batch_sz, src_len)
        attn_weight = (1 - p_gen) * attn_weight
        p_vocab = p_vocab.scatter_add(dim=1, index=src_vecs, src=attn_weight)
        return p_vocab

    def cal_loss(self, p_vocab, target, attn_weight, coverage_vec):
        """
        :param p_vocab: (batch_sz, extend_vocab_size)
        :param target: (batch_sz, 1)
        :param attn_weight: (batch_sz, src_len)
        :param coverage_vec: (batch_sz, src_len)
        :return loss: (batch_sz)
        """
        # (batch_sz, 1)
        target_prob = torch.gather(input=p_vocab, dim=1, index=target)
        mask = torch.ne(target, 0).float()
        # (batch_sz, 1)
        loss = -torch.log(target_prob + 1e-9)
        ## coverage mechanism
        cov_loss = torch.sum(torch.min(attn_weight, coverage_vec), dim=1)
        # (batch_sz)
        loss = (loss.squeeze(1) + self.lamb * cov_loss) * mask
        return loss


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
        :return dec_embed: (1, batch_sz, embed_sz)
        """
        # (batch_sz, 1) -> (batch_sz, 1, embed_sz)
        dec_embed = self.dropout(self.embed(dec_input))
        # (batch_sz, 1, hidden_sz), (1, batch_sz, hidden_sz)
        dec_output, dec_hidden = self.rnn(dec_embed, dec_hidden)
        # (batch_sz, 1, hidden_sz), (1, batch_sz, hidden_sz), (1, batch_sz, embed_sz)
        return dec_output, dec_hidden, dec_embed.transpose(0, 1)


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        # For encoder output
        self.Wh = nn.Linear(2 * hidden_size, hidden_size)
        # For decoder hidden state
        self.Ws = nn.Linear(hidden_size, hidden_size)
        # For coverage mechanism
        self.Wc = nn.Linear(1, hidden_size)
        self.V = nn.Linear(hidden_size, 1)

    def forward(self, enc_output, enc_mask, dec_hidden, coverage_vec):
        """
        :param enc_output: (batch_sz, src_len, 2*hidden_sz)
        :param enc_mask: (batch_sz, src_len, 1)
        :param dec_hidden: (1, batch_sz, hidden_sz)
        :param coverage_vec: (batch_sz, src_len, 1)
        :return context_vec: (batch_sz, 2*hidden_sz)
        :return attn_weight: (batch_sz, src_len)
        :return coverage_vec: (batch_sz, src_len, 1)
        """
        enc_mask = (1 - enc_mask).bool()
        src_len = enc_output.shape[1]
        # (1, batch_sz, hidden_sz) -> (batch_sz, 1, hidden_sz) -> (batch_sz, src_len, hidden_sz)
        dec_hidden = dec_hidden.transpose(0, 1).expand(-1, src_len, -1)
        # (batch_sz, src_len, hidden_sz)
        attn_input = self.Wh(enc_output) + self.Ws(dec_hidden) + self.Wc(coverage_vec)
        # (batch_sz, src_len, hidden_sz) -> (batch_sz, src_len, 1)
        attn_weight = self.V(torch.tanh(attn_input))
        attn_weight = F.softmax(attn_weight.masked_fill(enc_mask, float('-inf')), dim=1)
        # (batch_sz, (1, src_len)*(src_len, 2*hidden_sz)) -> (batch_sz, 1, 2*hidden_sz) -> (batch_sz, 2*hidden_sz)
        context_vec = torch.bmm(attn_weight.transpose(1, 2), enc_output).squeeze(1)
        # (batch_sz, src_len, 1)
        coverage_vec = coverage_vec + attn_weight
        # (batch_sz, 2*hidden_sz), (batch_sz, src_len), (batch_sz, src_len, 1)
        return context_vec, attn_weight.squeeze(2), coverage_vec


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
