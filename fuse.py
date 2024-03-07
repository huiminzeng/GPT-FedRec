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
                            
def save_llm_results(args, user_id, rerank_text, meta, answer):
    # user_id is not the true user id in the dataset, but the sample id without shuffle
    export_root = os.path.join('llm_results', args.dataset_code, 
                                'num_clients_' + str(args.num_clients), 
                                'samples_per_client_' + str(args.num_samples),
                                'lambda_' + str(args.lambda_ensemble))
    
    if not os.path.exists(export_root):
        os.makedirs(export_root)

    retrieved_data_path = os.path.join(export_root, 'rerank_'+ str(user_id) + '.pkl')

    data = {}
    data['rerank'] = rerank_text
    data['answer'] = meta[answer][0]
    data['answer_int'] = answer

    with open(retrieved_data_path, 'wb') as f:
        pickle.dump(data, f)

def main(args):
    e5_probs, _ = load_e5_candidates(args)
    lru_probs, _ = load_lru_candidates(args)

    seed_everything(args.seed)
    _, test_data, meta = dataloader_factory(args)
    test_dataset = E5TestDataset(args, test_data, args.bert_max_len)

    ensembled_probs = args.lambda_ensemble * e5_probs + (1 - args.lambda_ensemble) * lru_probs 
    rank = (-ensembled_probs).argsort(dim=1)

    count = 0
    for i, (seq, answer) in enumerate(test_dataset):
        topk = rank[i][:args.topk]
        if answer in topk: # this is for saving budget
            count += 1
            rerank_text = get_rerank(seq, topk, meta, args)
            save_llm_results(args, i, rerank_text, meta, answer)
        else:
            continue
    
    print(args.lambda_ensemble, count)
    print('='*50)

if __name__ == "__main__":
    args.model_code = 'e5'
    set_template(args)
    main(args)