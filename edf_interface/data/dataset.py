import os
from typing import Union, Optional, List, Tuple, Dict, Any, Iterable, TypeVar, Type, NamedTuple, Sequence

from beartype import beartype
import torch
from torchvision.transforms import Compose

from . import registered_datatype
from .base import DataAbstractBase
from .demo import Demo
from .io_utils import load_yaml


@beartype
def save_demos(demos: List[Demo], dir: str, mute: bool = False):
    if not os.path.exists(dir):
        os.makedirs(dir)
        
    with open(os.path.join(dir, "data.yaml"), 'w') as f:
        for i, demo in enumerate(demos):
            demo_dir = f"data/demo_{i}"
            demo.save(root_dir=os.path.join(dir, demo_dir))
            f.write("- path: \"" + demo_dir + "\"\n")
            f.write("  type: \"" + demo.__class__.__name__ + "\"\n")
    if not mute:
        print(f"saving demonstrations to {os.path.join(dir, demo_dir)}")

@beartype
def load_demos(dir: str, annotation_file = "data.yaml") -> List[Demo]:
    files = load_yaml(file_path=os.path.join(dir, annotation_file))

    demos: List[Demo] = []
    for file in files:
        type_ = file['type']
        assert hasattr(registered_datatype, type_), f"Unknown data type {type_}"
        type_ = getattr(registered_datatype, type_)
        assert issubclass(type_, DataAbstractBase), f"{type_}"
        demos.append(type_.load(os.path.join(dir, file['path'])))

    return demos

@beartype
class DemoDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir: str, 
                 annotation_file: str = "data.yaml", 
                 device: Union[str, torch.device] = 'cpu',
                 dtype: torch.dtype = torch.float32):
        self.device = torch.device(device)  
        self.dtype = dtype
        self.data: List[Demo] = [demo.to(device=self.device, dtype=self.dtype) for demo in load_demos(dir = dataset_dir, annotation_file=annotation_file)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    @staticmethod
    def save(demos: List[Demo], dir: str, mute: bool = False):
        save_demos(demos=demos, dir=dir, mute=mute)