def get_input_prompt(args, seq, meta):
    prompt = "query: "
    if args.dataset_code in ['beauty', 'games', 'auto', 'toys_new']:
        for item in seq:
            title = meta[item][0]

            category = meta[item][2]
            category = ', '.join(category.split(', ')[-2:])

            prompt += title
            prompt += ', a product about '

            prompt += category
            prompt += '. \n'

    if args.dataset_code in ['ml-100k']:
        for item in seq[-10:]:
            title = meta[item][0]

            category = meta[item][2]
            category = ', '.join(category.split(', ')[-2:])

            prompt += title

            prompt += category
            prompt += '. \n'
    
    return prompt

def get_target_prompt(args, item, meta):
    prompt = "passage: "
    
    try:
        title = meta[item][0]
    except:
        title = meta[item[0]][0]

    description = meta[item][1]
    category = meta[item][2]

    prompt += title
    if args.dataset_code in ['beauty', 'games', 'auto', 'toys_new']:
        prompt += ', a product about '

    elif args.dataset_code in ['ml-100k']:
        prompt += ', a movie about '

    prompt += category
    prompt += '. '
    
    return prompt