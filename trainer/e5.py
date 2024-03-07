from .utils import *
from .base import *
from .prompts import *

import torch
import torch.nn.functional as F

import numpy as np
from abc import *

class E5Trainer(BaseTrainer):
    def __init__(self, args, model, client_data, client_id, global_round, meta, E5TrainDataset, E5ValidDataset, collate_fn):
        super().__init__(args, model, client_id, global_round)
        # get dataloader
        self.meta = meta
        self.train_loader, self.val_loader = self.train_val_split(client_data, E5TrainDataset, E5ValidDataset, collate_fn)
        
        self.ce = torch.nn.CrossEntropyLoss()
        self.mr = torch.nn.MarginRankingLoss(self.args.margin)


    def calculate_loss(self, batch):
        input_prompts, target_prompts, negative_prompts = self.get_batch_prompts(batch, mode='train')

        # forward pass
        input_tokens = self.model.tokenizer(input_prompts, max_length=256, truncation=True, padding=True, return_tensors="pt")
        input_tokens = {k: v.cuda() for k, v in input_tokens.items()}
        outputs = self.model.model(**input_tokens)
        embeddings = average_pool(outputs.last_hidden_state, input_tokens['attention_mask'])
        embeddings = F.normalize(embeddings, dim=-1)

        # positive ground-truth
        target_tokens = self.model.tokenizer(target_prompts, max_length=256, truncation=True, padding=True, return_tensors="pt")
        target_tokens = {k: v.cuda() for k, v in target_tokens.items()}
        outputs = self.model.model(**target_tokens)
        embeddings_pos = average_pool(outputs.last_hidden_state, target_tokens['attention_mask'])
        embeddings_pos = F.normalize(embeddings_pos, dim=-1)

        # cosine between positive pairs
        positive_logit = torch.sum(embeddings * embeddings_pos, dim=-1, keepdim=True)

        ### negative ground-truth
        negative_logit = []
        for i in range(self.args.num_negatives):
            negative_tokens = self.model.tokenizer(list(negative_prompts[:, i]), max_length=256, truncation=True, padding=True, return_tensors="pt")
            negative_tokens = {k: v.cuda() for k, v in negative_tokens.items()}
            outputs = self.model.model(**negative_tokens)
            embeddings_neg = average_pool(outputs.last_hidden_state, negative_tokens['attention_mask'])
            embeddings_neg = F.normalize(embeddings_neg, dim=-1)

            # cosine between negative pairs
            negative_logit.append(torch.sum(embeddings * embeddings_neg, dim=-1, keepdim=True))
        
        negative_logit = torch.cat(negative_logit, dim=-1)

        logits = torch.cat([positive_logit, negative_logit], dim=1) / 0.01
        labels = torch.zeros(len(logits)).long().cuda()

        # cross entropy
        loss = self.ce(logits, labels)

        # # margin rank loss
        # # this loss is optional
        # loss_margin = 0
        # logits = F.softmax(logits, dim=-1)
        # for i in range(1,self.args.num_negatives):
        #     loss_margin += self.mr(logits[:,0], logits[:,i], torch.ones(len(logits)).long().cuda())
        #     loss += 0.1 * loss_margin
        
        return loss

    def calculate_metrics(self, batch):
        input_prompts, target_prompts, negative_prompts = self.get_batch_prompts(batch, mode='val')
        batch_size = len(input_prompts)

        input_tokens = self.model.tokenizer(input_prompts, max_length=256, truncation=True, padding=True, return_tensors="pt")
        input_tokens = {k: v.cuda() for k, v in input_tokens.items()}

        # forward pass
        outputs = self.model.model(**input_tokens)
        embeddings = average_pool(outputs.last_hidden_state, input_tokens['attention_mask'])
        embeddings = F.normalize(embeddings, dim=-1)

        target_tokens = self.model.tokenizer(target_prompts, max_length=256, truncation=True, padding=True, return_tensors="pt")
        target_tokens = {k: v.cuda() for k, v in target_tokens.items()}

        # positive ground-truth
        outputs = self.model.model(**target_tokens)
        embeddings_pos = average_pool(outputs.last_hidden_state, target_tokens['attention_mask'])
        embeddings_pos = F.normalize(embeddings_pos, dim=-1)

        negative_tokens = self.model.tokenizer(list(negative_prompts.reshape(-1)), max_length=256, truncation=True, padding=True, return_tensors="pt")
        negative_tokens = {k: v.cuda() for k, v in negative_tokens.items()}

        # negative ground-truth
        outputs = self.model.model(**negative_tokens)
        embeddings_neg = average_pool(outputs.last_hidden_state, negative_tokens['attention_mask'])
        embeddings_neg = F.normalize(embeddings_neg, dim=-1).reshape(batch_size, self.args.num_negatives, -1)

        # cosine between negative pairs
        negative_logit = torch.sum(embeddings.unsqueeze(1) * embeddings_neg, dim=-1)

        # cosine between positive pairs
        positive_logit = torch.sum(embeddings * embeddings_pos, dim=1, keepdim=True)
        
        scores = torch.cat([positive_logit, negative_logit], dim=1) / 0.01
        labels = torch.zeros(len(scores)).long().cuda()

        metrics = absolute_recall_mrr_ndcg_for_ks(scores, labels.view(-1), self.metric_ks)

        return metrics

    def train_val_split(self, client_data, E5TrainDataset, E5ValidDataset, collate_fn):
        train_data = client_data[0]
        val_data = client_data[1]

        train_dataset = E5TrainDataset(self.args, train_data, self.args.bert_max_len, self.args.sliding_window_size, self.meta)
        val_dataset = E5ValidDataset(self.args, train_data, val_data, self.args.bert_max_len, self.meta)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.train_batch_size,
                                                    shuffle=True, pin_memory=True, num_workers=self.args.num_workers,
                                                    collate_fn=collate_fn)
        
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.args.val_batch_size,
                                                    shuffle=False, pin_memory=True, num_workers=self.args.num_workers,
                                                    collate_fn=collate_fn)
        
        return train_loader, val_loader
    
    def get_batch_prompts(self, batch, mode):
        items_pool = list(range(1, self.args.num_items))
        
        input_prompts = []
        target_prompts = []
        negative_prompts = []

        seed = 0
        for seq, answer in zip(batch[0], batch[1]):
            input_text = get_input_prompt(self.args, seq, self.meta)
            target_text = get_target_prompt(self.args, answer, self.meta)

            seen_items = set(seq)
            seen_items.add(answer)
            negative_pool = set(items_pool) - seen_items
            negative_pool = list(negative_pool)
            if mode == 'train':
                negative_sample = np.random.choice(negative_pool, self.args.num_negatives, replace=False)
            elif mode == 'val':
                np.random.seed(seed)
                negative_sample = np.random.choice(negative_pool, self.args.num_negatives, replace=False)
                seed += 1
                
            temp_texts = []
            for neg_sam in negative_sample:
                negative_text = get_target_prompt(self.args, neg_sam, self.meta)
                temp_texts.append(negative_text)

            input_prompts.append(input_text)
            target_prompts.append(target_text)
            negative_prompts.append(temp_texts)

        negative_prompts = np.array(negative_prompts)

        return input_prompts, target_prompts, negative_prompts