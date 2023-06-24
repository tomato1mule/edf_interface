from typing import Union, Tuple, Optional
import torch
import torch.nn.functional as F
import torch_cluster
from edf_interface.data import PointCloud, SE3

def check_pcd_collision(x: Union[PointCloud, torch.Tensor], y: Union[PointCloud, torch.Tensor], r: float) -> bool:
    if isinstance(x, PointCloud):
        x = x.points
    if isinstance(y, PointCloud):
        y = y.points

    return torch_cluster.radius(x=x, y=y, r = r).any().item()


def pcd_energy(x: Union[PointCloud, torch.Tensor], y: Union[PointCloud, torch.Tensor], cutoff_r: float, grad: bool =True) -> Tuple[torch.Tensor, Optional[torch.Tensor], int]:
    if isinstance(x, PointCloud):
        x = x.points
    if isinstance(y, PointCloud):
        y = y.points

    radius_graph = torch_cluster.radius(x=x, y=y, r = cutoff_r)
    n_edge = radius_graph.shape[-1]

    if n_edge == 0:
        if grad is True:
            return torch.tensor(0., device=x.device), torch.zeros(3, device=x.device), n_edge
        else:
            return torch.tensor(0., device=x.device), None, n_edge

    x = x[radius_graph[1]].detach()
    y = y[radius_graph[0]].detach()

    if grad is True:
        dist_y = torch.zeros(3, requires_grad=True, device=x.device)
        y = y + dist_y

    energy = (cutoff_r/(x-y).norm(dim=-1, p=1)).sum(dim=-1)

    if grad is True:
        energy.backward()
        return energy.detach(), dist_y.grad.detach(), n_edge
    else:
        return energy.detach(), None, n_edge
    
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