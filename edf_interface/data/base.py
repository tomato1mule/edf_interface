from __future__ import annotations
import inspect
import os
import builtins
from typing import Union, Optional, List, Tuple, Dict, Any, Iterable, TypeVar, Type, NamedTuple, Sequence, Generic, _GenericAlias, get_origin
from typing_extensions import Self
from abc import ABCMeta, abstractmethod
import warnings

import yaml
from beartype import beartype
from beartype.door import is_bearable, die_if_unbearable
import torch


from . import registered_datatype

_bool = builtins.bool
_device = Union[torch.device, str]
_dtype = torch.dtype

@beartype
def hint_to_type(x: Union[_GenericAlias, type]):
    if isinstance(x, _GenericAlias):
        return get_origin(x)
    elif isinstance(x, type):
        return x
    else:
        raise ValueError(f"x should be either a _GenericAlias or a type instance.")


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
    @property
    @abstractmethod
    def data_args_type(self) -> Dict[str, type]:
        pass

    @property
    @abstractmethod
    def metadata_args(self) -> List[str]:
        pass

    @property
    @abstractmethod
    def device(self) -> torch.device:
        pass

    def _dump_metadata(self, x: Dict) -> str:
        return yaml.safe_dump(x)

    @property
    def metadata(self) -> Dict[str, Any]:
        output = {'__type__': self.__class__.__name__} # record type of this class
        for arg in self.metadata_args:
            assert isinstance(arg, str), f"argument ({arg}) in metadata_args must be of string type."
            assert arg not in self.data_args_type.keys(), f"Variable '{arg}' has duplicate declarations in data_args and metadata_args"
            assert arg != '__type__', f"Don't use '__type__' in metadata_args. It's reserved."
            try:
                obj = getattr(self, arg)
            except AttributeError:
                raise AttributeError(f"Cannot find metadata attribute {arg} in {self.__class__.__name__}")
            output[arg] = obj
        try:
            self._dump_metadata(output) # Just for type checking.
        except yaml.representer.RepresenterError:
            raise TypeError(f"Only python primitive types can be used for metadata. If you want to use non-primitive types, change 'yaml.safe_dump(x)' in self._dump_metadata() to 'yaml.dump(x)'")
        return output

    # def __init__(self):
    #     self.check_init_args()

    # def check_init_args(self, init_fn=None):
    #     if init_fn is None:
    #         init_fn = self.__init__
        
    #     for arg, val in self.data_args_type.items():
    #         assert arg not in self.metadata_args_type.keys(), f"data_arg {arg} is also defined in metadata_arg."
    #         assert isinstance(val, type), f"data_arg_type ({arg}: {str(val)}) must be an object type, not {type(val)}"
    #     for arg, val in self.metadata_args_type.items():
    #         assert isinstance(val, type), f"metadata_arg_type ({arg}: {str(val)}) must be an object type, not {type(val)}"
            
    #     args_hint = {}
    #     for key, val in inspect.signature(init_fn).parameters.items():
    #         assert key==val.name, f"Using variable args or kwargs ({val.name}) is not allowed in __init__."

    #         if key in self.data_args_type.keys() or key in self.metadata_args_type.keys():
    #             annotation = Any if val.annotation is inspect._empty else val.annotation
    #             # default_var = val.default
    #             args_hint[key] = annotation

    #     assert 'metadata' not in args_hint.keys(), f"Don't use 'metadata' as a parameter name! It is reserved."

    #     for arg, type_ in self.data_args_type.items():
    #         assert arg in args_hint.keys(), f"data_arg {arg} must be defined as a parameter of __init__()!"
    #         assert args_hint[arg] == type_, f"self.data_args_type[{arg}] = {self.data_args_type[arg]} != {args_hint[arg]}"
    #     for arg, type_ in self.metadata_args_type.items():
    #         assert arg in args_hint, f"metadata_arg_type {arg} must be defined as a parameter of __init__()!"
    #         assert args_hint[arg] == type_, f"self.metadata_args_type[{arg}] = {self.metadata_args_type[arg]} != {args_hint[arg]}"
            
    # @property
    # def metadata(self) -> Dict[str, Any]:
    #     metadata = {}
    #     for arg in self.metadata_args_type.keys():
    #         metadata[arg] = getattr(self, arg)
    #     return metadata

    def new(self, **kwargs) -> Self:
        """
        Returns a new object which is a shallow copy of original object, but with data and metadata that are specified as kwargs being replaced. 
        """
        for arg in (list(self.data_args_type.keys()) + list(self.metadata_args)):
            if arg not in kwargs.keys():
                kwargs[arg] = getattr(self, arg)

        return self.__class__(**kwargs)
    
    def to(self, *args, **kwargs) -> Self:
        """
        similar to pytorch Tensor objects' .to() method
        """
        input_dict = {}
        for arg in self.data_args_type.keys():
            obj = getattr(self, arg)
            if isinstance(obj, DataAbstractBase):
                obj = obj.to(*args, **kwargs)
                input_dict[arg] = obj
            elif isinstance(obj, torch.Tensor):
                assert '__tensor' not in kwargs.keys(), f"Don't use __tensor as a keyward arguments. It is reserved."
                obj = _torch_tensor_to(obj, *args, **kwargs)
                input_dict[arg] = obj
            elif hasattr(obj, 'to'):
                raise NotImplementedError(f"'to()' is not implemented for data type {type(obj)}")
                # input_dict[arg] = obj

        return self.new(**input_dict)
    
    def get_data_dict(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Returns recursive data dictionary.
        Similar to torch.nn.Module's .state_dict() method.
        """
        data_dict = {}
        for arg in self.data_args_type.keys():
            assert arg != '__type__', f"Don't use '__type__' in data_args. It's reserved."
            assert arg != 'metadata', f"Don't use 'metadata' in data_args. It's reserved."
            obj = getattr(self, arg)
            if isinstance(obj, DataAbstractBase):
                obj = obj.get_data_dict(*args, **kwargs)
            elif isinstance(obj, torch.Tensor):
                assert '__tensor' not in kwargs.keys(), f"Don't use __tensor as a keyward arguments. It's reserved."
                obj = _torch_tensor_to(obj, *args, **kwargs)
            data_dict[arg] = obj
        data_dict['metadata'] = self.metadata
        
        return data_dict
    
    @classmethod
    def from_data_dict(cls, data_dict: Dict[str, Any], *args, **kwargs) -> Self:
        """
        Reconstruct data object from dictionary.
        """
        inputs: Dict[str, Any] = {}
        for arg, val in data_dict.items():
            if arg == 'metadata':
                assert isinstance(val, Dict), f"data_dict['metadata'] must be a dictionary but {type(val)} is provided."
                assert cls.__name__ == val['__type__'], f"Class type {cls.__name__} does not match with type annotated in metadata ({val['__type__']})"
            else:
                assert arg in cls.data_args_type.keys(), f"Unknown data argument: {arg}"
                type_ = cls.data_args_type[arg]
                if issubclass(type_, DataAbstractBase):
                    assert isinstance(val, Dict), f"For arg of type {type(arg)}, data_dict[arg] must be a dictionary"
                    assert 'metadata' in val.keys(), f"For arg of type {type(arg)}, data_dict[arg] must be a dictionary, and has 'metadata' as a key"
                    assert type_.__name__ == val['metadata']['__type__'], f"{type_.__name__} != {val['metadata']['__type__']}"
                    val = type_.from_data_dict(data_dict=val, *args, **kwargs)
                else:
                    assert isinstance(val, type_), f"type({arg}) = {type(val)} != {type_}"
                    if isinstance(val, torch.Tensor):
                        val = _torch_tensor_to(__tensor=val, *args, **kwargs)
                inputs[arg] = val
        
        if 'metadata' in data_dict.keys():
            metadata = data_dict['metadata']
            assert isinstance(metadata, Dict)
            for arg in metadata.keys():
                assert arg not in inputs.keys(), f"metadata_arg {arg} already exists as a data argument!"
        else:
            metadata = {}
        
        input_kwargs = {}
        for k,v in {**inputs, **metadata}.items():
            if k=='__type__':  # __type__ is not required as an argument to the class constructor
                pass
            else:
                input_kwargs[k] = v
        return cls(**input_kwargs)
    
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
            repr = f"<{self.__class__.__name__}>  (device: {str(self.device)})\n"

        if not abbrv:
            repr += prefix + "Metadata: \n"
        # for arg in self.metadata_args:
        #     obj = getattr(self, arg)
        #     repr += bullet + f"{arg}: {obj.__repr__()}\n"
        for arg, obj in self.metadata.items():
            if arg == '__type__':
                pass
            else:
                repr += bullet + f"{arg}: {obj.__repr__()}\n"

        if not abbrv:
            repr += prefix + "Data: \n"
        for arg in self.data_args_type.keys():
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