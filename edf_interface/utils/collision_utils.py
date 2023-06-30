from typing import Union, Tuple, Optional
import torch
import torch.nn.functional as F
import torch_cluster
import torch_scatter
from edf_interface.data import transforms, se3, pcd_utils, PointCloud, SE3

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
def _pcd_energy(x: torch.Tensor, 
                       y: torch.Tensor, 
                       cutoff_r: float,
                       max_num_neighbor: int = 100,
                       eps: float = 0.001,
                       compute_grad: bool = True,
                       cluster_method: str = 'knn',
                       ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
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

    if cluster_method == 'radius':
        edges = torch_cluster.radius(
            x=x.detach(), 
            y=y.detach(), 
            r = cutoff_r, 
            max_num_neighbors=max_num_neighbor
        )
    elif cluster_method == 'knn':
        edges = torch_cluster.knn(
            x=x.detach(), 
            y=y.detach(), 
            k=max_num_neighbor,
        )
    else:
        raise ValueError(f"Unknown cluster method '{cluster_method}'")
    
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
    
# def pcd_energy(x: Union[PointCloud, torch.Tensor], y: Union[PointCloud, torch.Tensor], cutoff_r: float, grad: bool =True) -> Tuple[torch.Tensor, Optional[torch.Tensor], int]:
#     return _pcd_energy(x=convert_to_tensor(x), y=convert_to_tensor(y), cutoff_r=cutoff_r, grad=grad)

@torch.jit.script
def _se3_adjoint_lie_grad(Ts: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    """_summary_

    Args:
        Ts (torch.Tensor): (..., 7), (qw, qx, qy, qz, x, y, z)
        grad (torch.Tensor): (..., 6), (rx, ry, rz, vx, vy, vz)

    Returns:
        adjoint_grad (torch.Tensor): (..., 6), (rx, ry, rz, vx, vy, vz)

    Note:
    L_v f(g_0 g x) = L_{[Ad_g0]v} f(g g_0 x)
    => Grad_{g} f(g_0 g x) = Grad_{g} [Ad_g0]^{Transpose} f(g g_0 x)
    Note that gradient takes the transpose of adjoint matrix!!
    [Ad_T]^{Transpose} = [
        [R^{-1},   -R^{-1} skew(p)],
        [     0,        R^{-1}    ]
    ]
    """
    assert Ts.shape[-1] == 7, f"{Ts.shape}"
    assert grad.shape[-1] == 6, f"{grad.shape}"
    assert Ts.shape[:1] == grad.shape[:1], f"{Ts.shape}, {grad.shape}"

    qinv = transforms.quaternion_invert(Ts[..., :4]) # (..., 4)
    adj_grad_R = grad[..., :3] + torch.cross(Ts[..., 4:], grad[..., 3:]) # (..., 3)
    adj_grad_R = transforms.quaternion_apply(qinv, adj_grad_R) # (..., 3)
    adj_grad_v = transforms.quaternion_apply(qinv, grad[..., 3:]) # (..., 3)
    
    adj_grad = torch.cat([adj_grad_R, adj_grad_v], dim=-1) # (..., 6)

    return adj_grad


@torch.jit.script
def _optimize_pcd_collision_once(x: torch.Tensor, 
                                 y: torch.Tensor, 
                                 Ts: torch.Tensor,
                                 dt: float, 
                                 cutoff_r: float, 
                                 max_num_neighbors: int = 100,
                                 eps: float = 0.01,
                                 cluster_method: str = 'knn'):
    assert x.ndim == 2 and x.shape[-1] == 3, f"{x.shape}" # (nX, 3)
    assert y.ndim == 3 and y.shape[-1] == 3, f"{y.shape}" # (nPose, nY, 3)
    assert Ts.ndim == 2 and Ts.shape[-1] == 7, f"{Ts.shape}" # (nPose, 7)
    assert len(Ts) == len(y), f"{Ts.shape}, {y.shape}"
    n_poses, n_y_points = y.shape[:2]

    Ty = pcd_utils.transform_points(y, Ts, batched_pcd=True) # (nPose, nY, 3)
    energy, grad = _pcd_energy(
        x=x, 
        y=Ty, 
        cutoff_r=cutoff_r, 
        eps = eps, 
        max_num_neighbor=max_num_neighbors, 
        cluster_method=cluster_method
    ) # (nPose,), (nPose, 6)
    assert isinstance(grad, torch.Tensor)
    grad = _se3_adjoint_lie_grad(Ts, grad) # (nPose, 6)

    # disp = -grad / (grad.norm() + eps) * dt
    grad = grad * (cutoff_r**2)
    disp = -grad * dt
    disp_pose = se3._exp_map(disp) # (n_poses, 7)

    new_pose = se3._multiply(Ts, disp_pose)

    # done = torch.isclose(energy, torch.zeros_like(energy))

    return new_pose, energy
    
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