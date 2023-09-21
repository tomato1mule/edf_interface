from __future__ import annotations
import inspect
import os
import builtins
from typing import Union, Optional, List, Tuple, Dict, Any, Iterable, TypeVar, Type, NamedTuple, Sequence, Generic, _GenericAlias, get_origin
from typing_extensions import Self
from abc import ABCMeta, abstractmethod
import warnings
import pickle

import yaml
import gzip
from beartype import beartype
from beartype.door import is_bearable, die_if_unbearable
import torch


from . import registered_datatype
from .io_utils import pickle_serialize, pickle_deserialize


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

#@beartype
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
    
    def get_data_dict(self, *args, serialize=False, **kwargs) -> Dict[str, Any]:
        """
        Returns recursive data dictionary.
        Similar to torch.nn.Module's .state_dict() method.
        """
        kwargs['serialize'] = serialize

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
                if serialize:
                    obj = pickle_serialize(obj)
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
                    if isinstance(val, bytes):
                        val = pickle_deserialize(val)
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
    
    # def __repr__(self) -> str:
    #     return self.__class__.__name__

    def __repr__(self) -> str:
        #return f"<{self.__class__.__name__} (device: {str(self.device)})>"
        # return f"<{self.__class__.__name__} ({str(self.device)})>"
        device_str = str(self.device)
        if device_str != 'cpu':
            return f"<{self.__class__.__name__} ({device_str})>"
        else:
            return f"<{self.__class__.__name__}>"
    
    def __str__(self, abbrv: bool = False) -> str:
        if abbrv:
            prefix = ''
            bullet = '- '
        else:
            prefix = '  '
            bullet = prefix + '  - '            

        if abbrv:
            repr = ""
        else:
            repr = self.__repr__() + "\n"

        if not abbrv:
            repr += prefix + "Metadata: \n"
        # for arg in self.metadata_args:
        #     obj = getattr(self, arg)
        #     repr += bullet + f"{arg}: {obj.__str__()}\n"
        for arg, obj in self.metadata.items():
            if arg == '__type__':
                pass
            else:
                repr += bullet + f"{arg}: {obj.__str__()}\n"

        if not abbrv:
            repr += prefix + "Data: \n"
        for arg in self.data_args_type.keys():
            obj = getattr(self, arg)

            if isinstance(obj, DataAbstractBase):
                repr += bullet + f"{arg}: {obj.__repr__()}"
            else:
                repr += bullet + f"{arg}: <{type(obj).__name__}>"
            if hasattr(obj, 'shape'):
                repr += ' (Shape: ' + obj.shape.__str__() + ')\n'
                if abbrv:
                    subrepr = ''
                else:
                    subrepr: str = obj.__str__()
            elif isinstance(obj, DataAbstractBase):
                repr += '\n'
                if abbrv:
                    subrepr = ''
                else:
                    subrepr: str = obj.__str__(abbrv=True)
            else:
                repr += '\n'
                if abbrv:
                    subrepr = ''
                else:
                    subrepr: str = obj.__str__()
            
            # if abbrv:
            #     indent = ' ' * (len(bullet))
            # else:
            #     indent = ' ' * (len(bullet) + 4)
            # subrepr = subrepr.replace('\n', '\n' + indent)
            # subrepr += '\n'
            if abbrv:
                indent = ''
            else:
                indent = ' ' * (len(bullet) + 4)
                subrepr = subrepr.replace('\n', '\n' + indent)
                subrepr += '\n'

            repr += indent + subrepr

        return repr
    
    def save(self, root_dir: str, *args, **kwargs):
        if 'device' not in kwargs.keys():
            kwargs['device'] = 'cpu'

        if not os.path.exists(root_dir):
            os.makedirs(root_dir)

        with open(os.path.join(root_dir, 'metadata.yaml'), 'w') as f:
            f.write(self._dump_metadata(self.metadata))

        for arg in self.data_args_type.keys():
            obj = getattr(self, arg)
            dir = os.path.join(root_dir, arg)
            if isinstance(obj, DataAbstractBase):
                obj = obj.to(*args, **kwargs)
                obj.save(root_dir=dir)
            elif isinstance(obj, torch.Tensor):
                obj = obj.to(*args, **kwargs)
                torch.save(obj, dir + '.pt')
            else:
                with gzip.open(dir + 'gzip', 'wb') as f:
                    pickle.dump(obj, f)

    @classmethod
    def load(cls, root_dir: str) -> Self:
        from .io_utils import recursive_load_dict
        data_dict = recursive_load_dict(root_dir=root_dir)
        return cls.from_data_dict(data_dict)







#@beartype
class DataListAbstract(DataAbstractBase):
    metadata_args: List[str]
    data_seq: List[DataAbstractBase]
    _data_name_prefix: str = 'data_'
    
    @property
    def data_args_type(self) -> Dict[str, type]:
        outputs = {}
        for i, data in enumerate(self.data_seq):
            outputs[f"{self._data_name_prefix}{i}"] = type(data)
        return outputs

    def __len__(self) -> int:
        return len(self.data_seq)
    
    def __getitem__(self, idx) -> Union[Self, DataAbstractBase]:
        assert type(idx) == slice or type(idx) == int, "Indexing must be an integer or a slice with single axis."
        if type(idx) == int:
            return self.data_seq[idx]
        else:
            return self.new(data_seq=self.data_seq[idx])
        
    @classmethod
    def _get_data_idx(cls, name: str) -> Optional[int]:
        if name.startswith(cls._data_name_prefix):
            index = name.lstrip(cls._data_name_prefix)
            try:
                index = int(name.lstrip(cls._data_name_prefix))
                return index
            except ValueError:
                return None
        else:
            return None
    
    def __getattr__(self, name: str):
        data_idx: Optional[int] = self._get_data_idx(name=name)
        if data_idx is None:
            if hasattr(super(), '__getattr__'):
                return super().__getattr__(name=name)
            else:
                raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        else:
            return self[data_idx]
        
    def __setattr__(self, name: str, value: Any):
        data_idx: Optional[int] = self._get_data_idx(name=name)
        if data_idx is None:
            super().__setattr__(name, value)
        else:
            assert isinstance(value, DataAbstractBase), f"{name} must be an isntance of DataAbstractBase"
            if data_idx == len(self):
                self.data_seq.append(value)
            elif data_idx < len(self):
                self.data_seq[data_idx] = value
            else:
                raise IndexError(f"{name} index larger than maximum lenghth: {len(self)}")

    def new(self, **kwargs) -> Self:
        """
        Returns a new object which is a shallow copy of original object, but with data and metadata that are specified as kwargs being replaced. 
        """
        kwargs = kwargs.copy()
        for arg in (['data_seq'] + list(self.metadata_args)):
            if arg not in kwargs.keys():
                kwargs[arg] = getattr(self, arg)
        
        data_seq_dirty = False
        for key, val in kwargs.copy().items():
            if key.startswith(self._data_name_prefix):
                index = self._get_data_idx(key)
                if index is not None:
                    if index >= len(kwargs['data_seq']):
                        raise IndexError(f"Index {key} out of range (Max length: {len(kwargs['data_seq'])})")
                    else:
                        if not data_seq_dirty:
                            kwargs['data_seq'] = kwargs['data_seq'].copy()
                            data_seq_dirty = True
                        kwargs['data_seq'][index] = val
                        kwargs.pop(key)

        return self.__class__(**kwargs)
    
    @property
    def is_empty(self) -> bool:
        if len(self) == 0:
            return True
        else:
            return False
        
    def __init__(self, data_seq: Sequence[DataAbstractBase]):
        self.data_seq = data_seq
        super().__init__()
        
    @classmethod
    def empty(cls, *args, **kwargs) -> Self:
        return cls([], *args, **kwargs)

    @property
    def device(self) -> Optional[torch.device]:
        if self.is_empty:
            return None
        
        device = None
        for data in self.data_seq:
            if hasattr(data, 'device'):
                device = data.device
        
        return device
    
    def to(self, *args, **kwargs) -> Self:
        """
        similar to pytorch Tensor objects' .to() method
        """
        if self.is_empty:
            return self
        else:
            data_seq = []
            for data in self.data_seq:
                if isinstance(data, DataAbstractBase):
                    data = data.to(*args, **kwargs)
                    data_seq.append(data)
                elif isinstance(data, torch.Tensor):
                    assert '__tensor' not in kwargs.keys(), f"Don't use __tensor as a keyward arguments. It is reserved."
                    data = _torch_tensor_to(data, *args, **kwargs)
                    data_seq.append(data)
                elif hasattr(data, 'to'):
                    raise NotImplementedError(f"'to()' is not implemented for data type {type(data)}")
                else:
                    data_seq.append(data)

            return self.new(data_seq=data_seq)
    
    @classmethod
    def from_data_dict(cls, data_dict: Dict[str, Any], *args, **kwargs) -> Self:
        """
        Reconstruct data object from dictionary.
        """
        inputs: Dict[str, Any] = {}
        data_seq = []
        for arg, val in data_dict.items():
            if arg == 'metadata':
                assert isinstance(val, Dict), f"data_dict['metadata'] must be a dictionary but {type(val)} is provided."
                assert cls.__name__ == val['__type__'], f"Class type {cls.__name__} does not match with type annotated in metadata ({val['__type__']})"
            else:
                data_idx = cls._get_data_idx(arg)
                assert data_idx is not None, f"Variable name must be like {cls._data_name_prefix}_(idx)"
                assert data_idx == len(data_seq)

                assert isinstance(val, Dict), f"data_dict[{arg}] must be a dictionary"
                assert 'metadata' in val.keys(), f"data_dict[{arg}] must be a dictionary, and has 'metadata' as a key"
                
                from . import registered_datatype
                assert hasattr(registered_datatype, val['metadata']['__type__']), f"Unknown data type {val['metadata']['__type__']}"
                type_ = getattr(registered_datatype, val['metadata']['__type__'])
                assert issubclass(type_, DataAbstractBase), f"{type_}"

                val = type_.from_data_dict(data_dict=val, *args, **kwargs)
                data_seq.append(val)
        inputs['data_seq'] = data_seq

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
    





            

class Observation(DataAbstractBase):
    def __init__(self):
        super().__init__()

class Action(DataAbstractBase):
    def __init__(self):
        super().__init__()