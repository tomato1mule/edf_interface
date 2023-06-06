from __future__ import annotations
from typing import Union, Optional, List, Tuple, Dict, Any, Iterable, TypeVar, Type, NamedTuple, Sequence
import warnings
from beartype import beartype

import numpy as np
import torch


from .transforms import quaternion_apply, quaternion_multiply, axis_angle_to_quaternion, quaternion_invert, normalize_quaternion
from .base import DataAbstractBase, Action, Observation

class SE3(Action, Observation):
    @property
    def data_args_hint(self) -> Dict[str, type]:
        hint = {
            'poses': torch.Tensor,
        }
        return hint

    @property
    def metadata_args_hint(self) -> Dict[str, type]:
        hint = {
            'name': str
        }
        return hint


    def __init__(self, poses: torch.Tensor, name: str = '', device: Optional[Union[str, torch.device]] = None, renormalize: bool = True):
        self.name: str = name
        if not isinstance(poses, torch.Tensor):
            if device is None:
                device = torch.device('cpu')
            poses = torch.tensor(poses, device=device)

        assert poses.ndim <= 2 and poses.shape[-1] == 7

        if device is None:
            device = poses.device
        device = torch.device(device)
        if device != poses.device:
            poses = poses.to(device)
        
        if poses.ndim == 1:
            poses = poses.unsqueeze(-2)

        self.device = device
        self.poses = poses

        if not torch.allclose(self.poses[...,:4].detach().norm(dim=-1,keepdim=True), torch.tensor([1.0], device=device), rtol=0, atol=0.03):
            warnings.warn("SE3.__init__(): Input quaternion is not normalized")

        if renormalize:
            self.poses[...,:4] = self.poses[...,:4] / self.poses[...,:4].norm(dim=-1,keepdim=True)

        self.inv = self._inv

    @staticmethod
    def from_orn_and_pos(orns: torch.Tensor, positions: torch.Tensor, versor_last_input: bool = False, device: Optional[Union[str, torch.device]] = None) -> SE3:
        assert positions.ndim == 2 and orns.ndim == 2 and positions.shape[-1] == 3 and orns.shape[-1] == 4
        assert positions.shape[-2] == orns.shape[-2]

        if device is None:
            assert orns.device == positions.device
            device = positions.device
        else:
            device = torch.device(device)
            orns = orns.to(device)
            positions = positions.to(device)

        if versor_last_input:
            poses = torch.cat((orns[..., 3:4], orns[..., :3], positions), dim=-1)
        else:
            poses = torch.cat((orns, positions), dim=-1)

        return SE3(poses=poses, device=device)

    @staticmethod
    def from_numpy(orns: np.ndarray, positions: np.ndarray, versor_last_input: bool = False, device: Union[str, torch.device] = 'cpu') -> SE3:
        return SE3.from_orn_and_pos(orns = torch.tensor(orns, dtype=torch.float32, device=device), 
                                    positions=torch.tensor(positions, dtype=torch.float32, device=device), 
                                    versor_last_input=versor_last_input, 
                                    device=device)

    def __len__(self) -> int:
        return len(self.poses)

    def get_data_dict(self) -> Dict[str, torch.Tensor]:
        data_dict = {"poses": self.poses.detach().cpu()}
        return data_dict

    @staticmethod
    def from_data_dict(data_dict: Dict, device: Union[str, torch.device] = 'cpu') -> SE3:
        assert "poses" in data_dict.keys() 
        return SE3(poses=data_dict["poses"].to(device), device=device)
    
    @staticmethod
    def multiply(*Ts) -> SE3:
        T: SE3 = Ts[-1]
        q,x = T.poses[...,:4], T.poses[...,4:]
        q = q / q.norm(dim=-1, keepdim=True)

        for T in Ts[-2::-1]:
            assert len(T.poses) == len(q) == len(x) or len(T.poses) == 1 or len(q) == len(x) == 1
            x = quaternion_apply(T.poses[...,:4], x) + T.poses[...,4:]
            q = quaternion_multiply(T.poses[...,:4], q)
            q = q / q.norm(dim=-1, keepdim=True)
        return SE3(poses=torch.cat([q,x], dim=-1), renormalize=False)
    
    @staticmethod
    def inv(T) -> SE3:
        q, x = T.poses[...,:4], T.poses[...,4:]
        q_inv = quaternion_invert(q / q.norm(dim=-1, keepdim=True))
        q_inv = q_inv / q_inv.norm(dim=-1, keepdim=True)
        x_inv = -quaternion_apply(q_inv, x)
        return SE3(poses=torch.cat([q_inv, x_inv], dim=-1), renormalize=False)
    
    def _inv(self) -> SE3:
        return SE3.inv(self)
    
    def __mul__(self, other) -> SE3:
        return SE3.multiply(self, other)
    
    def __getitem__(self, idx) -> SE3:
        assert type(idx) == slice or type(idx) == int, "Indexing must be an integer or a slice with single axis."
        return SE3(poses=self.poses[idx], renormalize=False)
    
    @property
    def orns(self) -> torch.Tensor:
        return self.poses[...,:4]
    
    @property
    def points(self) -> torch.Tensor:
        return self.poses[...,4:]
    
    # @staticmethod
    # def apply(T: SE3, points: Union[torch.Tensor, PointCloud]):
    #     if type(points) == PointCloud:
    #         is_pcd = True
    #         pcd = points
    #         points = pcd.points
    #     else:
    #         is_pcd = False

    #     assert points.shape[-1] == 3, "points must be 3-dimensional."
    #     assert T.device == points.device, f"device of SE(3) ({T.device}) does not matches device of points ({points.device})."
    #     points = (quaternion_apply(T.orns, points.view(-1, 3)) + T.points).view(*(points.shape))
    #     if is_pcd:
    #         points = PointCloud(points=points, colors=pcd.colors)
    #     return points

    @staticmethod
    def empty(device: Union[str, torch.device] = 'cpu') -> SE3:
        return SE3(poses=torch.empty((0,7), device=device), device=device)
    
    def is_empty(self) -> bool:
        if len(self.poses) == 0:
            return True
        else:
            return False
