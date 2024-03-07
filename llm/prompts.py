import pdb

def get_prompts(seq, topk, meta, args):
    if args.dataset_code in ['beauty', 'games', 'auto', 'toys_new']:
        prompt = "I've purchased the following products in the past in order: \n"

        category_set = []
        part_history = seq[-5:]
        for i in range(len(part_history)):
            item = part_history[i]
            title = meta[item][0]
            category = meta[item][2]
            category = category.split(', ')
            category_set.extend(category)
            
            prompt += str(i+1)
            prompt += '. '
            prompt += title
            if i <= 8:
                prompt += ', \n'
            else:
                prompt += '. \n\n'

        category_set = list(set(category_set))
        prompt += ' The categories of these purchased products are '
        prompt += ', '.join(category_set)
        prompt += '. \n'

        prompt += 'Now there are 20 candidate products that I can consider to purchase next: \n'
        topk = topk.tolist()
        for i in range(args.topk):
            item = topk[i]
            title = meta[item][0]
            prompt += str(i+1)
            prompt += '. '
            prompt += title
            if i <= args.topk - 1:
                prompt += ', \n'
            else:
                prompt += '. \n'
    

    elif args.dataset_code in ['ml-100k']:
        prompt = "I've watched the following movies in the past in order: \n"

        category_set = []
        part_history = seq[-10:]
        for i in range(10):
            item = part_history[i]
            title = meta[item][0]
            category = meta[item][2]
            category = category.split(', ')
            category_set.extend(category)
            
            prompt += str(i+1)
            prompt += '. '
            prompt += title
            if i <= 8:
                prompt += ', \n'
            else:
                prompt += '. \n'

        category_set = list(set(category_set))
        prompt += ' The genres of these watched movies are '
        prompt += ', '.join(category_set)
        prompt += '. \n'

        prompt += 'Now there are 20 candidate movies that I can watch next: \n'
        topk = topk.tolist()
        for i in range(args.topk):
            item = topk[i]
            title = meta[item][0]
            prompt += str(i+1)
            prompt += '. '
            prompt += title
            if i <= args.topk - 1:
                prompt += ', \n'
            else:
                prompt += '. \n'

    return prompt