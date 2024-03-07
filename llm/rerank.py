import time
from fuzzywuzzy import fuzz

from .prompts import *
from .LLM import *

def get_rerank(seq, topk, meta, args):
    # convert seq into prompts
    prompts = get_prompts(seq, topk, meta, args)

    try:
        rerank_text = conversation(prompts, args)

    except:
        time.sleep(10)
        rerank_text = conversation(prompts, args)

    return rerank_text
    

def perform_rerank(rerank_text, topk, meta, flag):
    re_rank_list = rerank_text.split('\n')
    meta_reversed_keys = []
    for i in topk:
        meta_reversed_keys.append(meta[i][0])

    reranked_ids = []
    
    for item_text in re_rank_list:
        if '. ' in item_text:
            item = item_text.split('. ')[1]
            best_score = 0
            for key in meta_reversed_keys:
                score = fuzz.token_set_ratio(item, key)
                if score >= best_score:
                    best_score = score
                    best_match = key

            pos = meta_reversed_keys.index(best_match)
            reranked_ids.append(topk[pos])

    if flag:
        pdb.set_trace()
        
    return reranked_ids