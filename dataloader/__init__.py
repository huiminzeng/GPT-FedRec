from datasets import dataset_factory

from .lru import *
from .e5 import *

def dataloader_factory(args):
    dataset = dataset_factory(args)
    if args.model_code == 'lru':
        client_train, client_test = get_lru_data(args, dataset)
        return client_train, client_test

    elif args.model_code == 'e5':
        client_train, client_test, meta = get_e5_data(args, dataset)
        return client_train, client_test, meta