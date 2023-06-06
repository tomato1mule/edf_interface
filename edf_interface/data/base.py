from __future__ import annotations
import inspect
import os
import builtins
from typing import Union, Optional, List, Tuple, Dict, Any, Iterable, TypeVar, Type, NamedTuple, Sequence, Generic, _GenericAlias
from typing_extensions import Self
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
import warnings

from beartype import beartype
from beartype.door import is_bearable, die_if_unbearable
import torch


_bool = builtins.bool
_device = Union[torch.device, str]
_dtype = torch.dtype

@beartype
def _torch_tensor_to(__tensor: torch.Tensor, 
                     device: Optional[_device]=None, 
                     dtype: Optional[_dtype]=None, 
                     non_blocking: bool = False, 
                     copy: bool = False, 
                     *args, **kwargs) -> torch.Tensor:
    
    return __tensor.to(device=device, dtype=dtype, non_blocking=non_blocking, copy=copy)

@beartype
class DataAbstractBase(metaclass=ABCMeta):
    data_args_hint: Dict[str, type]
    metadata_args_hint: Dict[str, type]
    # other_args_hint: Dict[str, type]
    args_hint: Dict[str, Any]

    @property
    @abstractmethod
    def data_args_hint(self) -> Dict[str, type]:
        pass

    @property
    @abstractmethod
    def metadata_args_hint(self) -> Dict[str, type]:
        pass

    def __init__(self):
        self.check_args()

    def check_args(self):
        assert hasattr(self, 'data_args_hint'), f"'data_args_hint' must be explicitly specified."
        assert hasattr(self, 'metadata_args_hint'), f"'metadata_args_hint' must be explicitly specified."
        for arg, val in self.data_args_hint.items():
            assert arg not in self.metadata_args_hint.keys(), f"data_arg {arg} is also defined in metadata_arg."
            assert isinstance(val, type), f"Type hint ({arg}: {str(val)}) must be an object type, not {type(val)}"
        for arg, val in self.metadata_args_hint.items():
            assert isinstance(val, type), f"Type hint ({arg}: {str(val)}) must be an object type, not {type(val)}"
            
        

        self.args_hint = {}
        for key, val in inspect.signature(self.__init__).parameters.items():
            assert key==val.name, f"Using variable args or kwargs ({val.name}) is not allowed in __init__."
            annotation = Any if val.annotation is inspect._empty else val.annotation
            if annotation is Dict:
                annotation = dict
            elif annotation is List:
                annotation = list
            elif annotation is Tuple:
                annotation = tuple

            assert isinstance(annotation, type), f"Type hint ({key}: {str(annotation)}) must be an object type, not {type(annotation)}"
            default_var = val.default
            self.args_hint[key] = annotation

        assert 'metadata' not in self.args_hint.keys(), f"Don't use 'metadata' as a parameter name! It is reserved."

        for arg, hint in self.data_args_hint.items():
            assert arg in self.args_hint.keys(), f"data_arg {arg} must be defined as a parameter of __init__()!"
            assert self.args_hint[arg] == hint, f"self.data_args_hint[{arg}] = {self.data_args_hint[arg]} != {hint}"
        for arg, hint in self.metadata_args_hint.items():
            assert arg in self.args_hint, f"metadata_arg_hint {arg} must be defined as a parameter of __init__()!"
            assert self.args_hint[arg] == hint, f"self.metadata_args_hint[{arg}] = {self.metadata_args_hint[arg]} != {hint}"
            
    @property
    def metadata(self) -> Dict[str, Any]:
        metadata = {}
        for arg in self.metadata_args_hint.keys():
            metadata[arg] = getattr(self, arg)
        return metadata

    def new(self, **kwargs) -> Self:
        for arg in (list(self.data_args_hint.keys()) + list(self.metadata_args_hint.keys())):
            if arg not in kwargs.keys():
                kwargs[arg] = getattr(self, arg)

        return self.__class__(**kwargs)

    def _torch_tensor_to(self, device: Optional[_device]=None, 
                         dtype: Optional[_dtype]=None, 
                         non_blocking: bool = False, 
                         copy: bool = False, 
                         *args, **kwargs) -> Dict[str, torch.Tensor]:
        
        out_dict: Dict[str, torch.Tensor] = {}
        for arg in self.data_args_hint.keys():
            obj = getattr(self, arg)
            if isinstance(obj, torch.Tensor):
                obj = obj.to(device=device, dtype=dtype, non_blocking=non_blocking, copy=copy)
                out_dict[arg] = obj
        return out_dict
    
    def _data_to(self, *args, **kwargs) -> Dict[str, DataAbstractBase]:
        out_dict: Dict[str, DataAbstractBase] = {}
        for arg in self.data_args_hint.keys():
            obj = getattr(self, arg)
            if isinstance(obj, DataAbstractBase):
                obj = obj.to(*args, **kwargs)
                out_dict[arg] = obj
        return out_dict
    
    def to(self, *args, **kwargs) -> Self:
        out_dict = {**self._data_to(*args, **kwargs), **self._torch_tensor_to(*args, **kwargs)}
        return self.new(**out_dict)
    
    def get_data_dict(self, *args, **kwargs) -> Dict[str, Any]:
        data_dict = {}
        for arg in self.data_args_hint.keys():
            obj = getattr(self, arg)
            if isinstance(obj, DataAbstractBase):
                obj = obj.get_data_dict(*args, **kwargs)
            elif isinstance(obj, torch.Tensor):
                assert '__tensor' not in kwargs.keys(), f"Don't use __tensor as a keyward arguments. It is reserved."
                obj = _torch_tensor_to(__tensor = obj, *args, **kwargs)
            data_dict[arg] = obj
        data_dict['metadata'] = self.metadata
        
        return data_dict
    
    @classmethod
    def from_data_dict(cls, data_dict: Dict[str, Any], *args, **kwargs) -> Self:
        inputs: Dict[str, Any] = {}
        for arg, val in data_dict.items():
            if arg == 'metadata':
                continue
            else:
                assert arg in cls.data_args_hint.keys(), f"Unknown data argument: {arg}"
                hint = cls.data_args_hint[arg]
                if issubclass(hint, DataAbstractBase):
                    val = hint.from_data_dict(data_dict=val, *args, **kwargs)
                else:
                    assert isinstance(val, hint), f"type({arg}) = {type(val)} != {hint}"
                    if isinstance(val, torch.Tensor):
                        val = _torch_tensor_to(__tensor=val, *args, **kwargs)
                inputs[arg] = val
        
        if 'metadata' in data_dict.keys():
            metadata = data_dict['metadata']
            assert isinstance(metadata, Dict)
            for arg, val in metadata.items():
                assert arg not in inputs.keys(), f"metadata_arg {arg} already exists as a data argument!"
        else:
            metadata = {}
        
        inputs = {**inputs, **metadata}
        
        return cls(**inputs)
    
    def __repr__(self, abbrv: bool = False) -> str:
        if abbrv:
            prefix = ''
            bullet = '- '
        else:
            prefix = '  '
            bullet = prefix + '  - '            

        if abbrv:
            repr = ""
        else:
            repr = f"<{self.__class__.__name__}>\n"

        if not abbrv:
            repr += prefix + "Metadata: \n"
        for arg in self.metadata_args_hint.keys():
            obj = getattr(self, arg)
            repr += bullet + f"{arg}: {obj.__repr__()}\n"

        if not abbrv:
            repr += prefix + "Data: \n"
        for arg in self.data_args_hint.keys():
            obj = getattr(self, arg)

            repr += bullet + f"{arg}: <{type(obj).__name__}>"
            if hasattr(obj, 'shape'):
                repr += ' (Shape: ' + obj.shape.__repr__() + ')\n'
                if abbrv:
                    subrepr = ''
                else:
                    subrepr: str = obj.__repr__()
            elif isinstance(obj, DataAbstractBase):
                repr += '\n'
                if abbrv:
                    subrepr = ''
                else:
                    subrepr: str = obj.__repr__(abbrv=True)
            else:
                repr += '\n'
                if abbrv:
                    subrepr = ''
                else:
                    subrepr: str = obj.__repr__()
            
            if abbrv:
                indent = ' ' * (len(bullet))
            else:
                indent = ' ' * (len(bullet) + 4)
            subrepr = subrepr.replace('\n', '\n' + indent)
            subrepr += '\n'
            repr += indent + subrepr

        return repr
    
    def __str__(self) -> str:
        return self.__repr__()
            

class Observation(DataAbstractBase):
    def __init__(self):
        super().__init__()

class Action(DataAbstractBase):
    def __init__(self):
        super().__init__()