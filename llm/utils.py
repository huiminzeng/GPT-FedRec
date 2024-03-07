def get_message_list(args, prompts):
    if args.dataset_code in ['beauty', 'games', 'sports', 'auto', 'toys_new']:
        prompts += "Please rank these 20 products by measuring the possibilities that I would like to purchase next most, according to the given purchasing records. Please think step by step.\n"
        prompts += "Please show me your ranking results with order numbers. Split your output with line break. You MUST rank the given candidate products. You can not generate products that are not in the given candidate list.\n"

        message_list = [
                    {"role": "system", "content": "You are a helpful shopping assistant. Therefore, people might ask you to recommend products for them to buy."},
                    {"role": "user", "content": prompts},
                    ] 

    elif args.dataset_code in ['ml-100k']:
        prompts += "Please rank these 20 movies by measuring the possibilities that I would like to watch next most, according to my watching history. Please think step by step.\n"
        prompts += "Please show me your ranking results with order numbers. Split your output with line break. You MUST rank the given candidate movies. You can not generate movies that are not in the given candidate list.\n"
        message_list = [
                    {"role": "system", "content": "You are a movie fan and movie reviewer. Therefore, people might ask you to recommend movies."},
                    {"role": "user", "content": prompts},
                    ]
        
    print(prompts)
    return message_list