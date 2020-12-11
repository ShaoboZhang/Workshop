# -*- coding: utf-8 -*-
"""
@Author  : Shaobo Zhang
@Project : Workshop
@FileName: utils.py
@Discribe: 
"""

from tqdm import tqdm
from torch import optim
from time import sleep
import torch, os
import torch.nn.utils as tud


def train_eval(model, train_data, dev_data, args):
    def cal_loss(batch_data, is_train=True):
        # (batch_sz, src_len), (batch_sz, src_len, 1), (batch_sz, tgt_len)
        src_vecs, src_masks, tgt_vecs = [data.to(args.device) for data in batch_data[:3]]
        if args.pointer:
            oov_dicts = batch_data[-1]
            batch_loss = model(src_vecs, src_masks, tgt_vecs, oov_dicts) # (batch_sz)
        else:
            batch_loss = model(src_vecs, src_masks, tgt_vecs) # (batch_sz)
        return batch_loss if is_train else batch_loss.cpu().item()

    def evaluate():
        model.eval()
        eval_loss = eval_size = 0
        with torch.no_grad():
            for batch_data in dev_data:
                batch_size = len(batch_data)
                eval_size += batch_size
                batch_loss = cal_loss(batch_data, is_train=False)
                eval_loss += batch_loss * batch_size
        model.train()
        eval_loss /= eval_size
        return eval_loss

    # Make directory to save model
    model_dir = os.path.dirname(args.model_path)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    # Training model
    torch.cuda.empty_cache()
    model.to(args.device)
    model.train()
    print("开始训练...")
    sleep(0.5)
    # criterion = CrossEntropyLoss(ignore_index=0)    # ignore 'PAD' character
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    eval_loss = best_eval_loss = float('inf')
    for epoch in range(args.num_epochs):
        epoch_loss = epoch_size = 0
        with tqdm(train_data) as batch_progress:
            batch_progress.set_description(f"Epoch {epoch + 1}")
            for i, batch_data in enumerate(train_data):
                batch_size = len(batch_data)
                epoch_size += batch_size
                batch_loss = cal_loss(batch_data, is_train=True)
                optimizer.zero_grad()
                batch_loss.backward()
                tud.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                # update epoch loss
                batch_loss = batch_loss.cpu().item()
                epoch_loss += batch_loss * batch_size
                # evaluate model every few iters
                if (i + 1) % 80 == 0:
                    eval_loss = evaluate()
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        torch.save(model.state_dict(), args.model_path)
                batch_progress.set_postfix(Train_Loss=batch_loss, Val_Loss=eval_loss)
                batch_progress.update()
        batch_progress.set_postfix(Train_Loss=epoch_loss / epoch_size, Val_Loss=eval_loss)
        batch_progress.update()


