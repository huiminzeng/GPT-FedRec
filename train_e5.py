import os
import torch
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import wandb
import argparse

from config import *
from model import *
from dataloader import *
from trainer import *

from pytorch_lightning import seed_everything

def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def generate_candidates(model, test_data, meta, retrieved_data_path, args):
    # prepare test dataloader
    test_dataset = E5TestDataset(args, test_data, args.bert_max_len)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size,
                                                shuffle=False, pin_memory=True, num_workers=args.num_workers,
                                                collate_fn=collate_fn)

    model.eval()
    test_probs, test_labels = [], []
    with torch.no_grad():
        print('****************** Generating Candidates for Test Set ******************')
        candidate_embeddings = calculate_all_item_embeddings(model, meta, args)
        tqdm_dataloader = tqdm(test_loader)
        for batch_idx, batch in enumerate(tqdm_dataloader):
            scores, labels = calculate_logits(model, batch, meta, candidate_embeddings, args)
            test_probs.extend(scores.tolist())
            test_labels.extend(labels)
        test_metrics = absolute_recall_mrr_ndcg_for_ks(torch.tensor(test_probs), 
                                                    torch.tensor(test_labels).view(-1), args.metric_ks)
        print(test_metrics)

    with open(retrieved_data_path, 'wb') as f:
        pickle.dump({'test_probs': test_probs,
                    'test_labels': test_labels,
                    'test_metrics': test_metrics}, f)

def calculate_all_item_embeddings(model, meta, args):
    # preprare all item prompts
    candidate_prompts = []
    for item in range(1, args.num_items+1):
        candidate_text = get_target_prompt(args, item, meta)
        candidate_prompts.append(candidate_text)
    
    candidate_embeddings =[]
    
    with torch.no_grad():
        for i in tqdm(range(0, args.num_items, args.test_batch_size)):
            input_prompts = candidate_prompts[i: i + args.test_batch_size]
        
            input_tokens = model.tokenizer(input_prompts, max_length=256, truncation=True, padding=True, return_tensors="pt")
            input_tokens = {k: v.cuda() for k, v in input_tokens.items()}

            outputs = model.model(**input_tokens)
            embeddings = average_pool(outputs.last_hidden_state, input_tokens['attention_mask'])
            embeddings = F.normalize(embeddings, dim=-1)
            candidate_embeddings.append(embeddings)

        candidate_embeddings = torch.cat(candidate_embeddings)
        
    return candidate_embeddings

def calculate_logits(model, batch, meta, candidate_embeddings, args):
    seqs, labels = batch
    batch_size = len(seqs)

    input_prompts = get_batch_prompts(args, meta, batch)
    batch_size = len(input_prompts)

    input_tokens = model.tokenizer(input_prompts, max_length=256, truncation=True, padding=True, return_tensors="pt")
    input_tokens = {k: v.cuda() for k, v in input_tokens.items()}

    # forward pass
    outputs = model.model(**input_tokens)
    embeddings = average_pool(outputs.last_hidden_state, input_tokens['attention_mask'])
    embeddings = F.normalize(embeddings, dim=-1)

    seqs, labels = batch
    
    scores = torch.matmul(embeddings, candidate_embeddings.T)
    # 0 itme padding
    place_holder = torch.zeros((batch_size, 1)).cuda()
    scores = torch.cat([place_holder, scores], dim=-1)
    
    for i in range(batch_size):
        scores[i, seqs[i]] = -1e9
        scores[i, 0] = -1e9  # padding

    return scores, labels

def get_batch_prompts(args, meta, batch):
    
    input_prompts = []
    target_prompts = []
    for seq, answer in zip(batch[0], batch[1]):
        input_text = get_input_prompt(args, seq, meta)
        target_text = get_target_prompt(args, answer, meta)

        input_prompts.append(input_text)
        target_prompts.append(target_text)

    return input_prompts

def main(args, export_root=None):
    seed_everything(args.seed)
    client_data, test_data, meta = dataloader_factory(args)
    model = E5Model()
    if export_root == None:
        export_root = os.path.join(EXPERIMENT_ROOT, args.dataset_code, 
                                    'num_clients_' + str(args.num_clients), 
                                    'samples_per_client_' + str(args.num_samples),
                                    'e5')
    
    if not os.path.exists(export_root):
        os.makedirs(export_root)
    print("we are saving results to: ", export_root)

    model.cuda()

    # copy weights
    global_weights = model.state_dict()

    # Training
    for epoch in range(args.global_epochs):
        local_weights = []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        model.train()
        for idx in range(args.num_clients):
            local_model = E5Trainer(args=args, model=copy.deepcopy(model), client_data=client_data[idx], client_id=idx, global_round=epoch, meta=meta, 
                                    E5TrainDataset=E5TrainDataset, E5ValidDataset=E5ValidDataset, collate_fn=collate_fn)
            
            w = local_model.train()
            
            local_weights.append(copy.deepcopy(w))

        # update global weights
        global_weights = average_weights(local_weights)

        model.load_state_dict(global_weights)

    generate_candidates(model, test_data, meta, os.path.join(export_root, 'retrieved.pkl'), args)

    model_save_name = os.path.join(export_root, 'model.checkpoint')
    model_checkpoint = {'state_dict': model.state_dict()}
    torch.save(model_checkpoint, model_save_name)


if __name__ == "__main__":
    set_template(args)
    main(args, export_root=None)
