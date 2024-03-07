from .ml_100k import ML100KDataset
from .beauty import BeautyDataset
from .games import GamesDataset
from .auto import AutoDataset
from .toys_new import ToysNewDataset

DATASETS = {
    ML100KDataset.code(): ML100KDataset,
    BeautyDataset.code(): BeautyDataset,
    GamesDataset.code(): GamesDataset,
    AutoDataset.code(): AutoDataset,
    ToysNewDataset.code(): ToysNewDataset
}


def dataset_factory(args):
    dataset = DATASETS[args.dataset_code]
    return dataset(args)
