"""
@Author  : Shaobo Zhang
@Project : Workshop
@FileName: transformer_predict.py
"""

import torch
import numpy as np
from time import sleep
from tqdm import tqdm
from rouge import Rouge
from data.data_utils import Beam, id2output, replace_oov


class TransformerPredict:
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
        :return: Rouge scores
        """
        rouge = Rouge()
        f1_r1, f1_r2, f1_rl = [], [], []
        print("生成摘要...")
        sleep(0.5)
        self.model.eval()
        count = 0
        with torch.no_grad():
            for batch_data in tqdm(test_data):
                src_vec, src_mask, tgt_vec = [data.to(self.device) for data in batch_data[:3]]
                oov_dict = batch_data[-1] if len(batch_data) > 3 else None
                
                src_padding_mask = (src_vec == 0).to(self.device)  # (batch_sz, src_len)
                
                memory = self.model.encoder(src_vec, src_padding_mask)  # (batch_sz, src_len, hidden_sz)
                
                if beam_search:
                    summary = self.beam_search(memory, src_padding_mask, self.beam_width)
                else:
                    summary = self.beam_search(memory, src_padding_mask, 1)
                
                sentence = id2output(summary, self.vocab, oov_dict[0] if oov_dict else None)
                ref = id2output(tgt_vec.squeeze().cpu().tolist(), self.vocab, oov_dict[0] if oov_dict else None)
                score = rouge.get_scores(sentence, ref)[0]
                f1_r1.append(score['rouge-1']['f'])
                f1_r2.append(score['rouge-2']['f'])
                f1_rl.append(score['rouge-l']['f'])
                if (count + 1) % 100 == 0:
                    print('\r', ref)
                    print(sentence, '\n')
                count += 1
        return map(np.mean, (f1_r1, f1_r2, f1_rl))
    
    def beam_search(self, memory, src_padding_mask, beam_width=5):
        """
        If beam_width = 1, it's Greedy Search.
        If beam_width > 1, it's Beam Search.
        :param memory: (batch_sz, src_len, hidden_sz)
        :param src_padding_mask: (batch_sz, src_len)
        :param beam_width: number of beams
        :return: summary sentence
        """
        batch_size = memory.size(0)
        
        dec_input = torch.tensor([[self.vocab.BOS]], device=self.device)  # (1, 1)
        
        beams = [Beam(dec_input.squeeze(), 0, memory)]
        
        for i in range(self.max_len):
            new_beams = []
            eos_num = 0
            
            for beam in beams:
                if beam.seq_idx[-1] == self.vocab.EOS or len(beam.seq_idx) >= 50:
                    eos_num += 1
                    new_beams.append(beam)
                    if eos_num == beam_width:
                        return beams[0].seq_idx.cpu().tolist()
                    continue
                
                dec_input = beam.seq_idx.unsqueeze(0)  # (1, current_len)
                
                log_probs, _ = self.model.predict(dec_input, beam.dec_hidden, src_padding_mask)
                
                topk_probs, topk_idx = torch.topk(log_probs, k=beam_width)
                
                for prob, idx in zip(topk_probs, topk_idx):
                    idx = idx.reshape(1)
                    new_beams += [beam.extend(idx, prob.cpu().item(), memory)]
            
            new_beams = sorted(new_beams, reverse=True)
            beams = new_beams[:beam_width]
        
        return beams[0].seq_idx.cpu().tolist()
