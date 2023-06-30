from __future__ import annotations
from typing import Union, Optional, List, Tuple, Dict, Any, Iterable, TypeVar, Type, NamedTuple, Sequence
import warnings
from beartype import beartype

import numpy as np
import torch


from .transforms import quaternion_apply, quaternion_multiply, axis_angle_to_quaternion, quaternion_invert, normalize_quaternion, quaternion_to_matrix, se3_log_map, se3_exp_map, matrix_to_quaternion
from .base import DataAbstractBase, Action, Observation

@torch.jit.script
def _multiply(T1: torch.Tensor, T2: torch.Tensor) -> torch.Tensor:
    q, x = T2[...,:4], T2[...,4:]
    q = normalize_quaternion(q)

    assert len(T1) == len(q) == len(x) or len(T1) == 1 or len(q) == len(x) == 1
    x = quaternion_apply(T1[...,:4], x) + T1[...,4:]
    q = quaternion_multiply(T1[...,:4], q)
    q = normalize_quaternion(q)

    return torch.cat([q,x], dim=-1)

@torch.jit.script
def _inv(T: torch.Tensor) -> torch.Tensor:
    q, x = T[...,:4], T[...,4:]
    q_inv = quaternion_invert(normalize_quaternion(q))
    q_inv = normalize_quaternion(q_inv)
    x_inv = -quaternion_apply(q_inv, x)
    return torch.cat([q_inv, x_inv], dim=-1)

@torch.jit.script
def _log_map(poses: torch.Tensor) -> torch.Tensor:
    """
    returns Lie algebra for (Rx, Ry, Rz, Vx, Vy, Vz)
    """
    assert poses.shape[-1] == 7, f"{poses.shape}"
    R = quaternion_to_matrix(poses[..., :4])
    x = poses[..., 4:].unsqueeze(-1)
    T = torch.cat([R,x],dim=-1)
    T = torch.cat([T, torch.cat([torch.zeros_like(T[...,:1,:3]), torch.ones_like(T[...,:1,3:])], dim=-1)], dim=-2)

    L = se3_log_map(T) # (Vx, Vy, Vz, Rx, Ry, Rz)
    return torch.cat([L[...,3:], L[...,:3]], dim=-1) # (Rx, Ry, Rz, Vx, Vy, Vz)

@torch.jit.script
def _exp_map(lie: torch.Tensor) -> torch.Tensor:
    assert lie.shape[-1] == 6, f"{lie.shape}"
    lie = torch.cat([lie[...,3:], lie[...,:3]], dim=-1) # (Rx, Ry, Rz, Vx, Vy, Vz) -> (Vx, Vy, Vz, Rx, Ry, Rz)
    T = se3_exp_map(lie) # (..., 4, 4)
    q = matrix_to_quaternion(T[...,:3, :3]) # (n, 4)
    x = T[...,:3,-1] # (n, 3)
    T = torch.cat([q,x], dim=-1) # (n, 7)
    return T

@beartype
class SE3(Action, Observation):
    data_args_type: Dict[str, type] = {
        'poses': torch.Tensor,
    }

    metadata_args: List[str] = ['name', 'unit_length']

    poses: torch.Tensor
    name: str
    unit_length: str

    @property
    def device(self) -> torch.device:
        return self.poses.device

    def __init__(self, poses: torch.Tensor, name: str = '', unit_length: str = '1 [m]', renormalize: bool = True):
        super().__init__()
        self.name: str = name
        self.unit_length: str = unit_length
        assert poses.ndim <= 2 and poses.shape[-1] == 7

        if poses.ndim == 1:
            poses = poses.unsqueeze(-2)
        self.poses = poses

        if renormalize:
            self.poses[...,:4] = self.poses[...,:4] / self.poses[...,:4].norm(dim=-1,keepdim=True)

        if not torch.allclose(self.poses[...,:4].detach().norm(dim=-1,keepdim=True), torch.tensor([1.0], device=self.device, dtype=self.poses.dtype), rtol=0, atol=0.03):
            warnings.warn("SE3.__init__(): Input quaternion is not normalized")

        self.inv = self._i_n_v

    @staticmethod
    def from_orn_and_pos(orns: torch.Tensor, positions: torch.Tensor, versor_last_input: bool = False) -> SE3:
        assert orns.device == positions.device
        assert positions.ndim == 2 and orns.ndim == 2 and positions.shape[-1] == 3 and orns.shape[-1] == 4
        assert positions.shape[-2] == orns.shape[-2]

        if versor_last_input:
            poses = torch.cat((orns[..., 3:4], orns[..., :3], positions), dim=-1)
        else:
            poses = torch.cat((orns, positions), dim=-1)

        return SE3(poses=poses)

    @staticmethod
    def from_numpy(orns: np.ndarray, positions: np.ndarray, versor_last_input: bool = False, device: Union[str, torch.device] = 'cpu') -> SE3:
        return SE3.from_orn_and_pos(orns = torch.tensor(orns, dtype=torch.float32, device=device), 
                                    positions=torch.tensor(positions, dtype=torch.float32, device=device), 
                                    versor_last_input=versor_last_input)

    def __len__(self) -> int:
        return len(self.poses)
       
    @staticmethod
    def multiply(*Ts) -> SE3:
        Ts = [T.poses if isinstance(T, SE3) else T for T in Ts]

        T2: torch.Tensor = Ts[-1]

        for T in Ts[-2::-1]:
            T2 = _multiply(T, T2)
        return SE3(poses=T2, renormalize=False)
    
    @staticmethod
    def inv(T: Union[SE3, torch.Tensor]):
        if isinstance(T, SE3):
            T = T.poses
        return SE3(poses=_inv(T), renormalize=False)
    
    def _i_n_v(self) -> SE3:
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

    @staticmethod
    def empty(device: Union[str, torch.device] = 'cpu') -> SE3:
        return SE3(poses=torch.empty((0,7), device=device))
    
    @property
    def is_empty(self) -> bool:
        if len(self.poses) == 0:
            return True
        else:
            return False
