from .utils import *
from .base import *

import torch
import torch.nn as nn

from abc import *

import sys
sys.path.append('../dataloader')
from dataloader import *

class LRUTrainer(BaseTrainer):
    def __init__(self, args, model, client_data, client_id, global_round):
        super().__init__(args, model, client_id, global_round)
        self.ce = nn.CrossEntropyLoss(ignore_index=0)
        # get dataloader
        self.train_loader, self.val_loader = self.train_val_split(client_data)

    def calculate_loss(self, batch):
        batch = self.to_device(batch)

        seqs, labels = batch
        logits = self.model(seqs)
        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)
        loss = self.ce(logits, labels)
            
        return loss

    def calculate_metrics(self, batch, exclude_history=True):
        batch = self.to_device(batch)

        seqs, labels = batch
        scores = self.model(seqs)[:, -1, :]
        B, L = seqs.shape
        if exclude_history:
            for i in range(L):
                scores[torch.arange(scores.size(0)), seqs[:, i]] = -1e9
            scores[:, 0] = -1e9  # padding
        metrics = absolute_recall_mrr_ndcg_for_ks(scores, labels.view(-1), self.metric_ks)
        return metrics

    def train_val_split(self, client_data):
        train_data = client_data[0]
        val_data = client_data[1]

        train_dataset = LRUTrainDataset(self.args, train_data, self.args.bert_max_len, self.args.sliding_window_size)
        val_dataset = LRUValidDataset(self.args, train_data, val_data, self.args.bert_max_len)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.train_batch_size,
                                                    shuffle=True, pin_memory=True, num_workers=self.args.num_workers)
        
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.args.val_batch_size,
                                                    shuffle=False, pin_memory=True, num_workers=self.args.num_workers)
            
        return train_loader, val_loader