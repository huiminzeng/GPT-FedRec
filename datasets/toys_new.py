from .base import AbstractDataset
from .utils import *

from datetime import date
from pathlib import Path
import pickle
import shutil
import tempfile
import os

import pdb

import gzip
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

import pdb

class ToysNewDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'toys_new'

    def url(cls):
        pass
    
    def maybe_download_raw_dataset(self):
        pass

    def load_ratings(self):
        data_dir = self._get_rawdata_folder_path()
        lines = []
        with open(data_dir.joinpath('sequential_data.txt'), 'r') as f:
            for line in f:
                lines.append(line.rstrip('\n'))
        
        user_items = {}
        for line in lines:
            user, items = line.strip().split(' ', 1)
            items = items.split(' ')
            items = [int(item) for item in items]
            user_items[int(user)] = items

        return user_items
    
    def load_datamap(self):
        dataset_path = self._get_rawdata_folder_path()
        with open(dataset_path.joinpath('datamaps.json'), "r") as f:
            datamaps = json.load(f)
        
        user2id = datamaps['user2id']
        user2id = {k: int(v) for k, v in user2id.items()}
        
        item2id = datamaps['item2id']
        item2id = {k: int(v) for k, v in item2id.items()}

        id2item = datamaps['id2item']
        id2item = {int(k): v for k, v in id2item.items()}
        
        return user2id, item2id, id2item
    
    def preprocess(self):
        dataset_path = self._get_preprocessed_dataset_path()
        if dataset_path.is_file():
            print('Already preprocessed. Skip preprocessing')
            return
        if not dataset_path.parent.is_dir():
            dataset_path.parent.mkdir(parents=True)
        
        # load sequential data
        user_items = self.load_ratings()    # start with 1

        # load data map
        user2id, item2id, id2item = self.load_datamap()

        # load meta data
        meta = self.load_meta_dict(item2id) # converted id2meta

        # split train val test
        train, val, test = self.split(user_items)
        
        dataset = {'train': train,
                    'val': val,
                    'test': test,
                    'meta': meta,
                    'umap': user2id,
                    'smap': item2id}
        
        with dataset_path.open('wb') as f:
            pickle.dump(dataset, f)

    def split(self, user_items):
        print('Splitting')
        user_count = len(user_items)
        train, val, test = {}, {}, {}
        for i in range(1,user_count+1):
            items = user_items[i]
            train[i], val[i], test[i] = items[:-2], items[-2:-1], items[-1:]
        return train, val, test
    

    def load_ratings_df(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath(self.all_raw_file_names()[0])
        df = pd.read_csv(file_path, header=None)
        df.columns = ['uid', 'sid', 'rating', 'timestamp']
        return df
    
    def load_meta_dict(self, item2id):
        folder_path = self._get_rawdata_folder_path()

        def parse(path):
            g = gzip.open(path, 'r')
            for l in g:
                yield eval(l)   
        
        meta_dict = {}
        for meta in parse(folder_path.joinpath('meta.json.gz')):
            if meta['asin'] in item2id:
                if 'description' in meta.keys():
                    description = meta['description'].strip()
                    description = get_description(description)  # get partial descrption
                else:
                    description = 'There is no description for this item.'
                
                if 'title' in meta.keys():
                    t = meta['title'].strip()
                else:
                    t = 'There is no title for this item.'

                meta_dict[item2id[meta['asin']]] = [t, description, ', '.join(meta['categories'][0])] 

            else:
                continue

        return meta_dict

    
