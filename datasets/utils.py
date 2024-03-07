import numpy as np
import pandas as pd
from tqdm import tqdm
import urllib.request
from nltk.tokenize import sent_tokenize

from pathlib import Path
import zipfile
import tarfile
import sys
import pdb


def download(url, savepath):
    urllib.request.urlretrieve(url, str(savepath))
    print()


def unzip(zippath, savepath):
    print("Extracting data...")
    zip = zipfile.ZipFile(zippath)
    zip.extractall(savepath)
    zip.close()


def unziptargz(zippath, savepath):
    print("Extracting data...")
    f = tarfile.open(zippath)
    f.extractall(savepath)
    f.close()

def get_description(raw_discription):
    sentences = sent_tokenize(raw_discription)
    num_tokens = 0
    description = ''
    for sent in sentences:
        description += sent 
        description += ' '
        num_tokens += len(sent.split(' '))

        if num_tokens > 100:
            break
        
    return description