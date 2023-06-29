from typing import Union, Tuple, Optional
import torch
import torch.nn.functional as F
import torch_cluster
import torch_scatter
from edf_interface.data import PointCloud, SE3

def convert_to_tensor(x: Union[PointCloud, torch.Tensor]) -> torch.Tensor:
    if isinstance(x, PointCloud):
        return x.points
    else:
        assert isinstance(x, torch.Tensor)
        return x

@torch.jit.script
def _check_pcd_collision(x: torch.Tensor, y: torch.Tensor, r: float) -> torch.Tensor:
    assert x.ndim == 2 and x.shape[-1] == 3, f"{x.shape}" # (nX, 3)
    if y.ndim == 2:
        y = y.unsqueeze(0)
    assert y.ndim == 3 and y.shape[-1] == 3, f"{y.shape}" # (nPose, nY, 3)
    n_poses, n_y_points = y.shape[:2]
    y = y.view(-1,3).detach()
    x = x.detach()

    edges = torch_cluster.radius(x=x, y=y, r = r)
    edges_y_idx = edges[0]
    pose_idx = edges_y_idx // n_y_points

    n_edges = torch_scatter.scatter_sum(src = torch.ones_like(edges_y_idx), index = pose_idx, dim=-1, dim_size=n_poses) # (n_pose)
    
    return n_edges >= 1 # (nPoses,)

def check_pcd_collision(x: Union[PointCloud, torch.Tensor], y: Union[PointCloud, torch.Tensor], r: float) -> torch.Tensor:
    return _check_pcd_collision(x=convert_to_tensor(x), y=convert_to_tensor(y), r = r)

@torch.jit.script
def _pcd_energy_radius(x: torch.Tensor, 
                       y: torch.Tensor, 
                       cutoff_r: float,
                       max_num_neighbor: int = 100,
                       eps: float = 0.001,
                       compute_grad: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    assert x.ndim == 2 and x.shape[-1] == 3, f"{x.shape}" # (nX, 3)
    if y.ndim == 2:
        y = y.unsqueeze(0)
    assert y.ndim == 3 and y.shape[-1] == 3, f"{y.shape}" # (nPose, nY, 3)
    n_poses, n_y_points = y.shape[:2]

    x = x.detach() 
    y = y.detach() # (nPose, nY, 3)

    if compute_grad is True:
        trans_y = torch.zeros(n_poses, 3, device=x.device).requires_grad_(True)
        y = y + trans_y.unsqueeze(-2) # (nPose, nY, 3)

        rot_y = torch.zeros(n_poses, 3, device=x.device).requires_grad_(True)
        dR = rot_y.unsqueeze(-1) * torch.eye(3, device=x.device, dtype=x.dtype) # (n_poses, 3, 3)
        for i in range(3):
            rot_vec = dR[:, i:i+1, :] # (n_poses, 1, 3)
            y = y + torch.cross(rot_vec, y) # (nPose, nY, 3)
    else:
        rot_y, trans_y, dR, rot_vec = torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0), 
    
    y = y.view(-1,3)
    edges = torch_cluster.radius(
        x=x.detach(), 
        y=y.detach(), 
        r = cutoff_r, 
        max_num_neighbors=max_num_neighbor
    )
    edge_y_idx, edge_x_idx = edges[0], edges[1]

    if len(edge_y_idx) == 0:
        return torch.zeros(n_poses, device=x.device, dtype=x.dtype), torch.zeros(n_poses, 6, device=x.device, dtype=x.dtype)

    x = x[edge_x_idx]
    y = y[edge_y_idx]
    pose_idx = edge_y_idx // n_y_points

    energy = (cutoff_r/(torch.norm(x-y, dim=-1, p=1) + eps*cutoff_r)) # (n_edges)
    energy = torch_scatter.scatter_sum(src = energy, index = pose_idx, dim=-1, dim_size=n_poses) # (n_pose)

    if compute_grad is True:
        energy.sum().backward()
        return energy.detach(), torch.cat([rot_y.grad.detach(), trans_y.grad.detach()], dim=-1) # (n_poses,), (n_poses, 6)
    else:
        return energy.detach(), None
    
def pcd_energy(x: Union[PointCloud, torch.Tensor], y: Union[PointCloud, torch.Tensor], cutoff_r: float, grad: bool =True) -> Tuple[torch.Tensor, Optional[torch.Tensor], int]:
    return _pcd_energy(x=convert_to_tensor(x), y=convert_to_tensor(y), cutoff_r=cutoff_r, grad=grad)
    
def _optimize_pcd_collision_once(x: Union[PointCloud, torch.Tensor], y: Union[PointCloud, torch.Tensor], cutoff_r: float, dt: float, eps: float) -> Tuple[Union[PointCloud, torch.Tensor], SE3, bool]:
    if isinstance(x, PointCloud):
        x = x.points
    if isinstance(y, PointCloud):
        pcd = y
        y = y.points
    else:
        pcd = None

    energy, grad, n_edge = pcd_energy(x=x, y=y, cutoff_r=cutoff_r, grad=True)

    if n_edge == 0:
        done = True
    else:
        done = False

    disp = -grad / (grad.norm() + eps) * dt
    disp_pose = F.pad(F.pad(disp, pad=(3,0), value=0.), pad=(1,0), value=1.) # (1,0,0,0,x,y,z)
    disp_pose = SE3(disp_pose)

    if pcd is None:
        return y + disp, disp_pose, done
    else:
        return pcd.transformed(disp_pose)[0], disp_pose, done
    
def optimize_pcd_collision(x: Union[PointCloud, torch.Tensor], y: Union[PointCloud, torch.Tensor], cutoff_r: float, dt: float, eps: float, iters: int, rel_pose: Optional[SE3] = None) -> Tuple[Union[PointCloud, torch.Tensor], SE3]:
    if rel_pose is not None:
        if isinstance(y, PointCloud):
            y = y.transformed(Ts=rel_pose)[0]
        else:
            raise NotImplementedError

    Ts = []
    for _ in range(iters):
        y, T, done = _optimize_pcd_collision_once(x=x, y=y, cutoff_r=cutoff_r, dt=dt, eps=eps)
        Ts.append(T)
        if done:
            break

    if rel_pose is not None:
        Ts.append(rel_pose)
    return y, SE3.multiply(*Ts)