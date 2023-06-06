import os
import pickle

import gzip
import yaml

def load_yaml(file_path: str):
    """Loads yaml file from path."""
    with open(file_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config

def gzip_save(data, path: str):
    dir = os.path.dirname(path)

    if not os.path.exists(dir):
        os.makedirs(dir)

    with gzip.open(path, 'wb') as f:
        pickle.dump(data, f)

def gzip_load(path: str):
    with gzip.open(path, 'rb') as f:
        data = pickle.load(f)
    return data