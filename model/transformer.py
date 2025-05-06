"""
@Author  : Shaobo Zhang
@Project : Workshop
@FileName: transformer.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from data.data_utils import replace_oov


class TransformerModel(nn.Module):
    def __init__(self, args, vocab):
        super(TransformerModel, self).__init__()
        self.encoder = TransformerEncoder(
            args.vocab_size, 
            args.embed_size, 
            args.hidden_size, 
            args.num_heads, 
            args.num_layers, 
            args.enc_dropout
        )
        self.decoder = TransformerDecoder(
            args.vocab_size, 
            args.embed_size, 
            args.hidden_size, 
            args.num_heads, 
            args.num_layers, 
            args.dec_dropout
        )
        self.generator = nn.Linear(args.hidden_size, args.vocab_size)
        
        self.vocab = vocab
        self.device = args.device
        self.teacher_force = args.teacher_force
        
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src_vecs, src_mask, tgt_vecs, oov_dicts=None):
        """
        :param src_vecs: (batch_sz, src_len)
        :param src_mask: (batch_sz, 1, src_len)
        :param tgt_vecs: (batch_sz, tgt_len)
        :param oov_dicts: oov dictionaries (optional)
        :return: batch_loss
        """
        batch_size, tgt_len = tgt_vecs.shape
        
        if oov_dicts is not None:
            src_vecs = replace_oov(src_vecs, self.vocab, self.device)
        
        src_padding_mask = (src_vecs == 0).to(self.device)  # (batch_sz, src_len)
        tgt_padding_mask = (tgt_vecs == 0).to(self.device)  # (batch_sz, tgt_len)
        
        tgt_mask = self.generate_square_subsequent_mask(tgt_len).to(self.device)
        
        memory = self.encoder(src_vecs, src_padding_mask)  # (batch_sz, src_len, hidden_sz)
        
        output = self.decoder(
            tgt_vecs, 
            memory, 
            tgt_mask, 
            tgt_padding_mask, 
            src_padding_mask
        )  # (batch_sz, tgt_len, hidden_sz)
        
        logits = self.generator(output)  # (batch_sz, tgt_len, vocab_sz)
        
        shift_logits = logits[:, :-1, :].contiguous()  # (batch_sz, tgt_len-1, vocab_sz)
        shift_targets = tgt_vecs[:, 1:].contiguous()   # (batch_sz, tgt_len-1)
        
        loss_fn = nn.CrossEntropyLoss(ignore_index=0)
        loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_targets.view(-1))
        
        return loss
    
    def predict(self, dec_input, memory, src_padding_mask):
        """
        :param dec_input: (batch_sz, current_len)
        :param memory: (batch_sz, src_len, hidden_sz)
        :param src_padding_mask: (batch_sz, src_len)
        :return: log_probs, memory
        """
        tgt_len = dec_input.size(1)
        tgt_mask = self.generate_square_subsequent_mask(tgt_len).to(self.device)
        tgt_padding_mask = (dec_input == 0).to(self.device)
        
        output = self.decoder(
            dec_input, 
            memory, 
            tgt_mask, 
            tgt_padding_mask, 
            src_padding_mask
        )  # (batch_sz, tgt_len, hidden_sz)
        
        logits = self.generator(output[:, -1])  # (batch_sz, vocab_sz)
        log_probs = F.log_softmax(logits, dim=-1)
        
        return log_probs, memory
    
    def generate_square_subsequent_mask(self, sz):
        """Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_heads, num_layers, dropout):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=hidden_size,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, src_padding_mask):
        """
        :param src: (batch_sz, src_len)
        :param src_padding_mask: (batch_sz, src_len)
        :return: memory: (batch_sz, src_len, embed_sz)
        """
        src = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)  # (batch_sz, src_len, embed_sz)
        src = self.pos_encoder(src)  # (batch_sz, src_len, embed_sz)
        
        memory = self.transformer_encoder(src, src_key_padding_mask=src_padding_mask)  # (batch_sz, src_len, embed_sz)
        
        return memory


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_heads, num_layers, dropout):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        decoder_layers = nn.TransformerDecoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=hidden_size,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, tgt, memory, tgt_mask, tgt_padding_mask, memory_key_padding_mask):
        """
        :param tgt: (batch_sz, tgt_len)
        :param memory: (batch_sz, src_len, embed_sz)
        :param tgt_mask: (tgt_len, tgt_len)
        :param tgt_padding_mask: (batch_sz, tgt_len)
        :param memory_key_padding_mask: (batch_sz, src_len)
        :return: output: (batch_sz, tgt_len, embed_sz)
        """
        tgt = self.embedding(tgt) * math.sqrt(self.embedding.embedding_dim)  # (batch_sz, tgt_len, embed_sz)
        tgt = self.pos_encoder(tgt)  # (batch_sz, tgt_len, embed_sz)
        
        output = self.transformer_decoder(
            tgt, 
            memory, 
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )  # (batch_sz, tgt_len, embed_sz)
        
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        :param x: (batch_sz, seq_len, d_model)
        :return: (batch_sz, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
