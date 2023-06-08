import os
import pickle
from typing import Callable

import gzip
import yaml
import torch

from .base import DataAbstractBase

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

def recursive_load_dict(root_dir):
    data_dict = {}
    for _, dirs, files in os.walk(root_dir):
        break
    files.sort()
    dirs.sort()
    
    for file in files:
        filename, extension = file.split('.')
        file_path = os.path.join(root_dir, file)

        if extension == 'pt':
            data = torch.load(file_path)
        elif extension == 'yaml':
            with open(file_path) as f:
                data = yaml.load(f, Loader=yaml.FullLoader)
        elif extension == 'gzip':
            with gzip.open(file_path, 'rb') as f:
                data = pickle.load(f)
            return data
        else:
            raise ValueError(f"Unknown extension '{extension}' in '{file_path}'")
        data_dict[filename] = data

    for dir in dirs:
        data_dict[dir] = recursive_load_dict(os.path.join(root_dir, dir))

    return data_dict

def serialize(serializer):
    def _serialize(fn):
        def wrapped(*args, **kwargs):
            out = fn(*args, **kwargs)
            return serializer(out)
        return wrapped
    return _serialize

# def serialize_if_data():
#     def serializer(x):
#         if isinstance(x, DataAbstractBase):
#             return x.get_data_dict(serialize=True)
#         else:
#             return x
#     return serialize(serializer=serializer)

serialize_if_data = serialize(
    serializer = lambda x: x.get_data_dict(serialize=True) if isinstance(x, DataAbstractBase) else x
)

