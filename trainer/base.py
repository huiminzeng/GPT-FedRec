from model import *
from config import *
from .utils import *

import copy 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from abc import ABCMeta

class BaseTrainer(metaclass=ABCMeta):
    def __init__(self, args, model, client_id, global_round):
        self.args = args
        self.device = args.device
        self.model = model.to(self.device)

        self.local_epochs = args.local_epochs
        self.metric_ks = args.metric_ks
        self.best_metric = args.best_metric
        
        self.client_id = client_id
        self.global_round = global_round 

        # get optimizer
        self.optimizer = self._create_optimizer()

        if args.enable_lr_schedule:
            if args.enable_lr_warmup:
                self.lr_scheduler = self.get_linear_schedule_with_warmup(
                    self.optimizer, args.warmup_steps, len(self.train_loader) * self.num_epochs)
            else:
                self.lr_scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer, step_size=args.decay_step, gamma=args.gamma)

    def train(self):
        accum_iter = 0
        self.best_metric = 0
        self.best_model = None

        if self.global_round == 0:
            self.best_metric = self.validate()
            self.best_model = copy.deepcopy(self.model.cpu()).state_dict()

        for epoch in range(self.local_epochs):  
            self.train_one_epoch(epoch, accum_iter)
            torch.cuda.empty_cache()

        return self.best_model

    def train_one_epoch(self, epoch, accum_iter):
        average_meter_set = AverageMeterSet()
        tqdm_dataloader = tqdm(self.train_loader)
        self.model.cuda()
        for batch_idx, batch in enumerate(tqdm_dataloader):
            self.model.train()

            self.optimizer.zero_grad()
            loss = self.calculate_loss(batch)
            loss.backward()
            self.clip_gradients(self.args.max_grad_norm)
            self.optimizer.step()
            if self.args.enable_lr_schedule:
                self.lr_scheduler.step()

            average_meter_set.update('loss', loss.item())
            tqdm_dataloader.set_description(
                'Global Epoch {}, Client ID {}, Epoch {}, loss {:.3f} '.format(self.global_round, self.client_id,
                                                                            epoch+1, average_meter_set['loss'].avg)) 
            if self.args.model_code == 'e5':
                if batch_idx % 40 == 0:
                    current_metric = self.validate() 
                    if current_metric >= self.best_metric:
                        self.best_metric = current_metric
                        self.best_model = copy.deepcopy(self.model.cpu()).state_dict()
                        self.model.cuda()

        if self.args.model_code == 'lru':
            current_metric = self.validate() 
            if current_metric >= self.best_metric:
                self.best_metric = current_metric
                self.best_model = copy.deepcopy(self.model.cpu()).state_dict()
                self.model.cuda()            

    def validate(self):
        self.model.eval()
        average_meter_set = AverageMeterSet()
        with torch.no_grad():
            tqdm_dataloader = tqdm(self.val_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                metrics = self.calculate_metrics(batch)
                self._update_meter_set(average_meter_set, metrics)
                self._update_dataloader_metrics(tqdm_dataloader, average_meter_set)

        current_metric = average_meter_set.sums()["NDCG@1"] + average_meter_set.sums()["Recall@1"]

        return current_metric
    
    def to_device(self, batch):
        return [x.to(self.device) for x in batch]

    # @abstractmethod
    def calculate_loss(self, batch):
        pass
    
    # @abstractmethod
    def calculate_metrics(self, batch):
        pass
    
    def clip_gradients(self, limit=1.0):
        nn.utils.clip_grad_norm_(self.model.parameters(), limit)

    def _update_meter_set(self, meter_set, metrics):
        for k, v in metrics.items():
            meter_set.update(k, v)

    def _update_dataloader_metrics(self, tqdm_dataloader, meter_set):
        description_metrics = ['NDCG@%d' % k for k in self.metric_ks[:3]] + ['Recall@%d' % k for k in self.metric_ks[:3]]
        description = 'Eval: ' + ', '.join(s + ' {:.4f}' for s in description_metrics)
        description = description.replace('NDCG', 'N').replace('Recall', 'R')
        description = description.format(*(meter_set[k].avg for k in description_metrics))
        tqdm_dataloader.set_description(description)

    def _create_optimizer(self):
        args = self.args
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'layer_norm']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': args.weight_decay,
            },
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.},
        ]
        if args.optimizer.lower() == 'adamw':
            return optim.AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
        elif args.optimizer.lower() == 'adam':
            return optim.Adam(optimizer_grouped_parameters, lr=args.lr, weight_decay=args.weight_decay)
        else:
            raise NotImplementedError

    def get_linear_schedule_with_warmup(self, optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
            )

        return LambdaLR(optimizer, lr_lambda, last_epoch)
    
    