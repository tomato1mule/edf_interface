from typing import Optional, Callable
import pickle

import torch

from edf_interface.data import DataAbstractBase
from edf_interface.data.io_utils import pickle_serialize, pickle_deserialize
from edf_interface.pyro import utils
from edf_interface.pyro import expose as expose_

def default_serializer(x):
    if isinstance(x, DataAbstractBase):
        x = x.get_data_dict(serialize=True)
    elif isinstance(x, torch.Tensor):
        x = pickle_serialize(x)
    else:
        pass
    return x

def serialize_input(serializer: Callable = default_serializer) -> Callable:
    return utils.serialize_input(serializer=serializer)

def serialize_output(serializer: Callable = default_serializer) -> Callable:
    return utils.serialize_output(serializer=serializer)

def expose(serializer: Callable = default_serializer) -> Callable:
    def wrapped(fn: Callable):
        return expose_(serialize_output(serializer=serializer)(fn))
    return wrapped