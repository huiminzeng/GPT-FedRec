from .base import AbstractDataloader
from .split_client import split_clients
import os
import torch
import random
import pickle
import numpy as np
import torch.utils.data as data_utils

import pdb

def worker_init_fn(worker_id):
    random.seed(np.random.get_state()[1][0] + worker_id)                                                      
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def get_lru_data(args, dataset):
    dataset = dataset.load_dataset()
    train_data = dataset['train']
    val_data = dataset['val']
    test_data = dataset['test']
    umap = dataset['umap']
    smap = dataset['smap']
    user_count = len(umap)
    item_count = len(smap)
    
    client_train, client_test = split_clients(args.num_clients, args.num_samples, train_data, val_data, test_data, user_count)
    args.num_users = user_count
    args.num_items = item_count
    
    return client_train, client_test


class LRUTrainDataset(data_utils.Dataset):
    def __init__(self, args, u2seq, max_len, sliding_size):
        self.args = args
        self.max_len = max_len
        self.sliding_step = int(sliding_size * max_len)
        self.num_items = args.num_items
        
        assert self.sliding_step > 0
        self.all_seqs = []
        for u in sorted(u2seq.keys()):
            seq = u2seq[u]
            if len(seq) < self.max_len + self.sliding_step:
                self.all_seqs.append(seq)
            else:
                start_idx = range(len(seq) - max_len, -1, -self.sliding_step)
                self.all_seqs = self.all_seqs + [seq[i:i + max_len] for i in start_idx]

    def __len__(self):
        return len(self.all_seqs)

    def __getitem__(self, index):
        seq = self.all_seqs[index]
        labels = seq[-self.max_len:]
        tokens = seq[:-1][-self.max_len:]

        mask_len = self.max_len - len(tokens)
        tokens = [0] * mask_len + tokens

        mask_len = self.max_len - len(labels)
        labels = [0] * mask_len + labels

        return torch.LongTensor(tokens), torch.LongTensor(labels)


class LRUValidDataset(data_utils.Dataset):
    def __init__(self, args, u2seq, u2answer, max_len):
        self.args = args
        self.u2seq = u2seq
        self.u2answer = u2answer
        users = sorted(self.u2seq.keys())
        self.users = [u for u in users if len(u2answer[u]) > 0]
        self.max_len = max_len

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user]
        answer = self.u2answer[user]

        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq

        return torch.LongTensor(seq), torch.LongTensor(answer)


class LRUTestDataset(data_utils.Dataset):
    def __init__(self, args, u2seq, max_len):
        self.args = args
        self.u2seq = u2seq
        users = sorted(self.u2seq.keys())
        self.users = [u for u in users if len(u2seq[u]) > 0]
        self.max_len = max_len

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq, answer = self.u2seq[user]

        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq

        return torch.LongTensor(seq), torch.LongTensor(answer)