import numpy as np
import random
import torch
import argparse

RAW_DATASET_ROOT_FOLDER = 'data'
EXPERIMENT_ROOT = 'experiments'
STATE_DICT_KEY = 'model_state_dict'
OPTIMIZER_STATE_DICT_KEY = 'optimizer_state_dict'

def set_template(args):
    if args.dataset_code == 'ml-100k':
        args.bert_max_len = 200
    else:
        args.bert_max_len = 50

    if 'llm' in args.model_code: 
        batch = 16 if args.dataset_code == 'ml-100k' else 12
        args.lora_micro_batch_size = batch
    else: 
        batch = 16 if args.dataset_code == 'ml-100k' else 8

    args.train_batch_size = batch
    args.val_batch_size = batch
    args.test_batch_size = batch

    if torch.cuda.is_available(): args.device = 'cuda'
    else: args.device = 'cpu'
    args.optimizer = 'AdamW'
    args.weight_decay = 0.01
    args.enable_lr_schedule = True
    args.decay_step = 10000
    args.gamma = 1.
    args.enable_lr_warmup = False
    args.warmup_steps = 100

    args.metric_ks = [1, 5, 10, 20, 50]
    args.rerank_metric_ks = [1, 5, 10]
    args.best_metric = 'Recall@10'
    args.rerank_best_metric = 'NDCG@10'

    args.bert_num_blocks = 2
    args.bert_num_heads = 2
    args.bert_head_size = None


parser = argparse.ArgumentParser()

################
# Dataset
################
parser.add_argument('--dataset_code', type=str, default=None)
parser.add_argument('--min_rating', type=int, default=0)
parser.add_argument('--min_uc', type=int, default=5)
parser.add_argument('--min_sc', type=int, default=5)
parser.add_argument('--seed', type=int, default=42)

################
# Dataloader
################
parser.add_argument('--train_batch_size', type=int, default=64)
parser.add_argument('--val_batch_size', type=int, default=64)
parser.add_argument('--test_batch_size', type=int, default=64)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--sliding_window_size', type=float, default=1.0)
parser.add_argument('--negative_sample_size', type=int, default=10)

################
# Trainer
################
# optimization #
parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])

parser.add_argument('--optimizer', type=str, default='AdamW', choices=['AdamW', 'Adam'])
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--adam_epsilon', type=float, default=1e-9)
parser.add_argument('--momentum', type=float, default=None)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--max_grad_norm', type=float, default=5.0)
parser.add_argument('--enable_lr_schedule', type=bool, default=True)
parser.add_argument('--decay_step', type=int, default=10000)
parser.add_argument('--gamma', type=float, default=1)
parser.add_argument('--enable_lr_warmup', type=bool, default=False)
parser.add_argument('--warmup_steps', type=int, default=100)

parser.add_argument('--margin', type=float, default=0.1)

################
# FL
################
parser.add_argument('--num_clients', type=int, default=None)
parser.add_argument('--num_samples', type=int, default=None)
parser.add_argument('--num_negatives', type=int, default=75)
parser.add_argument('--global_epochs', type=int, default=5)
parser.add_argument('--local_epochs', type=int, default=50)

################
# Evaluation
################
parser.add_argument('--metric_ks', nargs='+', type=int, default=[1, 5, 10, 20, 50])
parser.add_argument('--best_metric', type=str, default='Recall@10')

################
# Retriever Model
################
parser.add_argument('--model_code', type=str, default=None)
parser.add_argument('--bert_max_len', type=int, default=50)
parser.add_argument('--bert_hidden_units', type=int, default=64)
parser.add_argument('--bert_num_blocks', type=int, default=2)
parser.add_argument('--bert_num_heads', type=int, default=2)
parser.add_argument('--bert_head_size', type=int, default=32)
parser.add_argument('--bert_dropout', type=float, default=0.2)
parser.add_argument('--bert_attn_dropout', type=float, default=0.2)
parser.add_argument('--bert_mask_prob', type=float, default=0.25)

################
# Inference Ensemble
################
parser.add_argument('--threshold', type=float, default=0.5)
parser.add_argument('--ensembled_tokp', type=int, default=20)
parser.add_argument('--lambda_ensemble', type=float, default=0.2)
parser.add_argument('--topk', type=int, default=20)

################
# OpenAI token
################
parser.add_argument('--openai_token', type=str, default=None)

args = parser.parse_args()
