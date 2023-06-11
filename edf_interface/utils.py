from typing import Optional, Callable
import pickle
import inspect

import torch

from edf_interface.data import DataAbstractBase
from edf_interface.data.io_utils import pickle_serialize, pickle_deserialize
from edf_interface.pyro import utils as pyro_utils, expose as pyro_expose

def default_serializer(x):
    # if isinstance(x, DataAbstractBase):
    #     x = x.get_data_dict(serialize=True)
    # elif isinstance(x, torch.Tensor):
    #     x = pickle_serialize(x)
    # else:
    #     pass
    # return x
    return pickle_serialize(x)

def default_deserializer(x):
    return pickle_deserialize(x)


def _expose(serializer: Callable = default_serializer,
           deserializer: Callable = default_deserializer,
           class_method: bool = True) -> Callable:
    def wrapped(fn: Callable):
        return pyro_expose(
            pyro_utils.deserialize_input(deserializer=deserializer, class_method=class_method)(
                pyro_utils.serialize_output(serializer=serializer)(
                    fn
                )
            )
        )
    return wrapped

def _wrap_remote(serializer: Callable = default_serializer,
                deserializer: Callable = default_deserializer,
                class_method: bool = False) -> Callable:
    def wrapped(fn: Callable):
        return pyro_utils.serialize_input(serializer=serializer, class_method=class_method)(
            pyro_utils.deserialize_output(deserializer=deserializer)(
                fn
            )
        )
    return wrapped

expose = _expose()
wrap_remote = _wrap_remote()