import os
import torch
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from config import *
from model import *
from dataloader import *
from trainer import *
from llm import *

from pytorch_lightning import seed_everything

def load_e5_candidates(args):
    load_root = os.path.join(EXPERIMENT_ROOT, args.dataset_code, 
                            'num_clients_' + str(args.num_clients), 
                            'samples_per_client_' + str(args.num_samples),
                            'e5')
    
    retrieved_data_path = os.path.join(load_root, 'retrieved.pkl')

    with open(retrieved_data_path, 'rb') as f:
        results = pickle.load(f)

    test_probs = results['test_probs']
    test_labels = results['test_labels']
    
    test_probs = torch.softmax(torch.tensor(test_probs) / 0.01, dim=-1)
    return test_probs, test_labels


def load_lru_candidates(args):
    load_root = os.path.join(EXPERIMENT_ROOT, args.dataset_code, 
                            'num_clients_' + str(args.num_clients), 
                            'samples_per_client_' + str(args.num_samples),
                            'lru')

    retrieved_data_path = os.path.join(load_root, 'retrieved.pkl')

    with open(retrieved_data_path, 'rb') as f:
        results = pickle.load(f)

    test_probs = results['test_probs']
    test_labels = results['test_labels']

    test_probs = torch.softmax(torch.tensor(test_probs), dim=-1)
    return test_probs, test_labels

def load_llm_results(file_name):
    id = int(file_name.split('/')[-1].split('.')[0].split('_')[-1])
    with open(file_name, 'rb') as f:
        data = pickle.load(f)

    rerank_text = data['rerank']
    answer = data['answer']
    answer_int = data['answer_int']

    return rerank_text, answer, answer_int, id

def eval(rank, labels, ks):
    metrics = {}
    labels = F.one_hot(labels, num_classes=rank.size(1))
    answer_count = labels.sum(1)

    labels_float = labels.float()

    len_seq = rank.shape[-1]

    cut = rank
    for k in sorted(ks, reverse=True):
        if len_seq >= k:
            cut = cut[:, :k]
            hits = labels_float.gather(1, cut)
            metrics['Recall@%d' % k] = \
                (hits.sum(1) / torch.min(torch.Tensor([k]).to(labels.device), labels.sum(1).float())).mean().cpu().item()
            
            metrics['MRR@%d' % k] = \
                (hits / torch.arange(1, k+1).unsqueeze(0).to(
                    labels.device)).sum(1).mean().cpu().item()

            position = torch.arange(2, 2+k)
            weights = 1 / torch.log2(position.float())
            dcg = (hits * weights.to(hits.device)).sum(1)
            idcg = torch.Tensor([weights[:min(int(n), k)].sum()
                                for n in answer_count]).to(dcg.device)
            ndcg = (dcg / idcg).mean()
            metrics['NDCG@%d' % k] = ndcg.cpu().item()
            
        else:
            continue

    return metrics


def main(args):
    e5_probs, e5_labels = load_e5_candidates(args)
    lru_probs, lru_labels = load_lru_candidates(args)
    
    seed_everything(args.seed)
    _, test_data, meta = dataloader_factory(args)
    test_dataset = E5TestDataset(args, test_data, args.bert_max_len)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size,
                                                shuffle=False, pin_memory=True, num_workers=args.num_workers,
                                                collate_fn=collate_fn)

    # naive baseline
    rank = (-lru_probs).argsort(dim=1)
    test_metrics_lru = eval(rank, torch.tensor(lru_labels).view(-1), args.metric_ks)
    
    # LRU + E5
    ensembled_probs = args.lambda_ensemble * e5_probs + (1 - args.lambda_ensemble) * lru_probs 
    rank = (-ensembled_probs).argsort(dim=1)
    test_metrics_e5 = eval(rank, torch.tensor(lru_labels).view(-1), args.metric_ks)
    
    # LR5 + E5 + LLM
    load_root = os.path.join("llm_results", args.dataset_code, 
                            'num_clients_' + str(args.num_clients), 
                            'samples_per_client_' + str(args.num_samples),
                            'lambda_' + str(args.lambda_ensemble))

    rerank_results = os.listdir(load_root)    
    wrong_counter = 0
    for file_name in rerank_results:
        rerank_text, answer, answer_int, id = load_llm_results(os.path.join(load_root, file_name))
        topk = rank[id][:args.topk].tolist()
        reranked_topk = perform_rerank(rerank_text, topk, meta, False)

        if len(topk) == len(reranked_topk) and (answer_int in reranked_topk):
            rank[id][:args.topk] = torch.tensor(reranked_topk).long()

        elif len(topk) >= len(reranked_topk) and (answer_int in reranked_topk):
            topk_set = set(topk)
            reranked_topk_set = set(reranked_topk)
            reranked_topk = reranked_topk + list(topk_set - reranked_topk_set)
            rank[id][:args.topk] = torch.tensor(reranked_topk[:args.topk]).long()

        else:
            continue
    
    test_metrics_llm = eval(rank, torch.tensor(lru_labels).view(-1), args.metric_ks)

    for k in [5,10]:
        recall_lru = test_metrics_lru['Recall@%d' % k]
        recall_e5 = test_metrics_e5['Recall@%d' % k]
        recall_llm = test_metrics_llm['Recall@%d' % k]

        ndcg_lru = test_metrics_lru['NDCG@%d' % k]
        ndcg_e5 = test_metrics_e5['NDCG@%d' % k]
        ndcg_llm = test_metrics_llm['NDCG@%d' % k]

        print('Recall@{} LRU: {:.4f}, E5: {:.4f}, LLM: {:.4f}'.format(k, recall_lru, recall_e5, recall_llm))
        print('NDCG@{} LRU: {:.4f}, E5: {:.4f}, LLM: {:.4f}'.format(k, ndcg_lru, ndcg_e5, ndcg_llm))

        print("="*50)
        
if __name__ == "__main__":
    args.model_code = 'e5'
    set_template(args)
    main(args)