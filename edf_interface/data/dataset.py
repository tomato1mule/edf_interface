import os
from typing import Union, Optional, List, Tuple, Dict, Any, Iterable, TypeVar, Type, NamedTuple, Sequence

from beartype import beartype
import torch
from torchvision.transforms import Compose

from .demo import DemoSequence
from .io_utils import load_yaml

@beartype
def save_demos(demos: List[DemoSequence], dir: str):
    if not os.path.exists(dir):
        os.makedirs(dir)
        
    with open(os.path.join(dir, "data.yaml"), 'w') as f:
        for i, demo in enumerate(demos):
            demo_dir = f"data/demo_{i}"
            demo.save(root_dir=os.path.join(dir, demo_dir))
            f.write("- \"" + demo_dir + "\"\n")

@beartype
def load_demos(dir: str, annotation_file = "data.yaml") -> List[DemoSequence]:
    files = load_yaml(file_path=os.path.join(dir, annotation_file))

    demos: List[DemoSequence] = []
    for file in files:
        demos.append(DemoSequence.load(os.path.join(dir, file)))

    return demos


class DemoSeqDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir: str, 
                 annotation_file: str = "data.yaml", 
                 load_transforms: Optional[Union[Compose, torch.nn.Module]] = None, 
                 transforms: Optional[Union[Compose, torch.nn.Module]] = None, 
                 device: Union[str, torch.device] = 'cpu'):
        device = torch.device(device)
        if device != torch.device('cpu'):
            #raise NotImplementedError
            pass
        
        self.device = device
        self.load_transforms = load_transforms if load_transforms else lambda x:x
        self.transforms = transforms if transforms else lambda x:x

        self.data: List[DemoSequence] = [self.load_transforms(demo).to(self.device) for demo in load_demos(dir = dataset_dir, annotation_file=annotation_file)]

    def __len__(self):
        return len(self.data)

    # def __getitem__(self, idx):
    #     data = self.data[idx]
    #     return {'raw': data, 'processed': self.transforms(data)}

    def __getitem__(self, idx):
        data = self.transforms(self.data[idx])
        return data