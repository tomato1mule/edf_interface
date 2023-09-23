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
from .visualize import visualize_pose


@beartype
class Demo(DataAbstractBase): 
    def __init__(self):
        super().__init__()

@beartype
class TargetPoseDemo(Demo):
    data_args_type: Dict[str, type] = {
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
        super().__init__()
        
        self.name = name
        if device is not None:
            scene_pcd = scene_pcd.to(device)
            grasp_pcd = grasp_pcd.to(device)
            target_poses = target_poses.to(device)

        assert scene_pcd.device == target_poses.device == grasp_pcd.device

        self.scene_pcd: PointCloud = scene_pcd
        self.target_poses: SE3 = target_poses
        self.grasp_pcd: PointCloud = grasp_pcd

    def plotly(self, point_size=3.0, width=800, height=800, ranges=None, bg_color = None):
        _, fig = visualize_pose(
            scene_pcd=self.scene_pcd,
            grasp_pcd=self.grasp_pcd,
            poses=self.target_poses,
            point_size=point_size, width=width, height=height,
            ranges=ranges
        )
        if bg_color is not None:
            from .visualize import update_background_color
            fig = update_background_color(fig, color=bg_color)
        return fig
    
    def show(self, point_size=3.0, width=800, height=800, ranges=None, bg_color = None):
        return self.plotly(point_size=point_size, width=width, height=height, ranges=ranges, bg_color=bg_color)

@beartype
class DemoSequence(DataListAbstract, Demo):
    metadata_args: List[str] = ['name']
    data_seq: List[Demo]
    _data_name_prefix: str = 'step_'

    def __init__(self, data_seq: List[Demo], name: str = ''):
        self.name = name
        super().__init__(data_seq=data_seq)
    
