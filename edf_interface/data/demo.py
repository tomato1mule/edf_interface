from __future__ import annotations
import os

from typing import Union, Optional, List, Tuple, Dict, Any, Iterable, TypeVar, Type, NamedTuple, Sequence
from abc import ABCMeta, abstractmethod
import warnings
from beartype import beartype

import torch
from torchvision.transforms import Compose

from .pointcloud import PointCloud
from .se3 import SE3
from .base import DataAbstractBase, Observation, Action, DataListAbstract



class Demo(DataAbstractBase): 
    def __init__(self):
        super().__init__()

class TargetPoseDemo(Demo):
    data_args_hint: Dict[str, type] = {
        'scene_pcd': PointCloud,
        'grasp_pcd': PointCloud,
        'target_poses': SE3,
    }

    metadata_args: List[str] = ['name']

    scene_pcd: PointCloud
    grasp_pcd: PointCloud
    target_poses: SE3
    name: str

    @property
    def device(self) -> torch.device:
        return self.scene_pcd.device

    def __init__(self, target_poses: SE3 = SE3.empty(),
                 scene_pcd: PointCloud = PointCloud.empty(), 
                 grasp_pcd: PointCloud = PointCloud.empty(),
                 name: str = '',
                 device: Optional[Union[str, torch.device]] = None):
        super.__init__()
        
        self.name = name
        if device is not None:
            scene_pcd = scene_pcd.to(device)
            grasp_pcd = grasp_pcd.to(device)
            target_poses = target_poses.to(device)

        assert scene_pcd.device == target_poses.device == grasp_pcd.device

        self.scene_pcd: PointCloud = scene_pcd
        self.target_poses: SE3 = target_poses
        self.grasp_pcd: PointCloud = grasp_pcd

class DemoSequence(DataListAbstract):
    metadata_args: List[str] = ['name']
    data_seq: List[Demo]
    _data_name_prefix: str = 'demo_'

    def __init__(self, data_seq: List[Demo], name: str = ''):
        self.name = name
        super().__init__(data_seq=data_seq)



    

# def save_demos(demos: List[DemoSequence], dir: str):
#     if not os.path.exists(dir):
#         os.makedirs(dir)
        
#     with open(os.path.join(dir, "data.yaml"), 'w') as f:
#         for i, demo in enumerate(demos):
#             data_dir = "data"
#             filename = f"demo_{i}.gzip"
#             demo.save_data(path=os.path.join(dir, data_dir, filename))
#             f.write("- \""+os.path.join(data_dir, filename)+"\"\n")

# def load_demos(dir: str, annotation_file = "data.yaml") -> List[DemoSequence]:
#     files = load_yaml(file_path=os.path.join(dir, annotation_file))

#     demos: List[DemoSequence] = []
#     for file in files:
#         demos.append(DemoSequence.from_file(os.path.join(dir, file)))

#     return demos


# class DemoSeqDataset(torch.utils.data.Dataset):
#     def __init__(self, dataset_dir: str, 
#                  annotation_file: str = "data.yaml", 
#                  load_transforms: Optional[Union[Compose, torch.nn.Module]] = None, 
#                  transforms: Optional[Union[Compose, torch.nn.Module]] = None, 
#                  device: Union[str, torch.device] = 'cpu'):
#         device = torch.device(device)
#         if device != torch.device('cpu'):
#             #raise NotImplementedError
#             pass
        
#         self.device = device
#         self.load_transforms = load_transforms if load_transforms else lambda x:x
#         self.transforms = transforms if transforms else lambda x:x

#         self.data: List[DemoSequence] = [self.load_transforms(demo).to(self.device) for demo in load_demos(dir = dataset_dir, annotation_file=annotation_file)]

#     def __len__(self):
#         return len(self.data)

#     # def __getitem__(self, idx):
#     #     data = self.data[idx]
#     #     return {'raw': data, 'processed': self.transforms(data)}

#     def __getitem__(self, idx):
#         data = self.transforms(self.data[idx])
#         return data
    
