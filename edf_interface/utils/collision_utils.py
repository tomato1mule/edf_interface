from typing import Union, Tuple, Optional, List
from beartype import beartype
import torch
import torch.nn.functional as F
import torch_cluster
import torch_scatter
from edf_interface.data import transforms, se3, pcd_utils, PointCloud, SE3, preprocess

def convert_to_tensor(x: Union[PointCloud, SE3, torch.Tensor]) -> torch.Tensor:
    if isinstance(x, PointCloud):
        return x.points
    elif isinstance(x, SE3):
        return x.poses
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
    r = torch.norm(x-y, dim=-1, p=1)

    cutoff = True
    if cutoff and cluster_method == 'knn':
        inrange_edge_mask = r<=cutoff_r
        r = r[inrange_edge_mask]
        pose_idx = pose_idx[inrange_edge_mask]
    else:
        inrange_edge_mask = None

    energy = (cutoff_r/(r + eps*cutoff_r))   # (n_edges)
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
    adj_grad_R = grad[..., :3] - torch.cross(Ts[..., 4:], grad[..., 3:]) # (..., 3)
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
    # done = torch.isclose(energy, torch.zeros_like(energy))
    assert isinstance(grad, torch.Tensor)
    grad = _se3_adjoint_lie_grad(Ts, grad) # (nPose, 6)

    # disp = -grad / (grad.norm() + eps) * dt
    grad = grad * (torch.tensor([1., 1., 1., cutoff_r, cutoff_r, cutoff_r], device=grad.device, dtype=grad.dtype))
    disp = -grad * dt * cutoff_r

    # -------------------------------------------------------------------------------------------------- #
    # If gradient is not detached due to torch.jit bug, use with torch.no_grad() context.
    # -------------------------------------------------------------------------------------------------- #

    # disp_pose = se3._exp_map(disp) # (n_poses, 7)
    # new_pose = se3._multiply(Ts, disp_pose)

    # If gradient is not detached due to torch.jit bug, use the following instead.
    with torch.no_grad():
        disp_pose = se3._exp_map(disp) # (n_poses, 7)
        new_pose = se3._multiply(Ts, disp_pose)
    
    # -------------------------------------------------------------------------------------------------- #

    return new_pose, energy

@torch.jit.script
def _optimize_pcd_collision_trajectory(x: torch.Tensor, 
                                       y: torch.Tensor, 
                                       Ts: torch.Tensor, 
                                       n_steps: int,
                                       dt: float,
                                       cutoff_r: float,
                                       max_num_neighbors: int = 100,
                                       eps: float = 0.01,
                                       cluster_method: str = 'knn',
                                       revert_order: bool = False) -> torch.Tensor:
    """_summary_

    Args:
        x (torch.Tensor): (nX, 3)
        y (torch.Tensor): (nPose, nY, 3) or (nY, 3)
        Ts (torch.Tensor): (nPose, 7)
        n_steps (int): _description_
        dt (float): _description_
        cutoff_r (float): _description_
        max_num_neighbors (int, optional): _description_. Defaults to 100.
        eps (float, optional): _description_. Defaults to 0.01.
        cluster_method (str, optional): _description_. Defaults to 'knn'.

    Returns:
        trajectories: _description_
    """
    assert n_steps >= 1
    assert Ts.ndim == 2 and Ts.shape[-1] == 7, f"{Ts.shape}"
    n_poses = len(Ts)
    assert x.ndim == 2 and x.shape[-1] == 3, f"{x.shape}"
    assert (y.ndim == 2 or y.ndim == 3) and y.shape[-1] == 3, f"{y.shape}"
    if y.ndim == 2:
        y = y.expand(n_poses, -1, 3)

    trajectories = [Ts]
    for i in range(n_steps-1):
        new_pose, energy = _optimize_pcd_collision_once(x=x, y=y, Ts=trajectories[-1], dt=dt, cutoff_r=cutoff_r, max_num_neighbors=max_num_neighbors, eps=eps, cluster_method=cluster_method)
        trajectories.append(new_pose)
    trajectories = torch.stack(trajectories, dim=0) # (n_steps, n_poses, 7)
    trajectories = trajectories.movedim(0, -2) # (n_poses, n_steps, 7)
    if revert_order:
        trajectories = torch.flip(trajectories[...,::], dims=(-2,)) # (n_poses, n_steps, 7)

    return trajectories

@beartype
def optimize_pcd_collision_trajectory(x: Union[PointCloud, torch.Tensor], 
                                      y: Union[PointCloud, torch.Tensor], 
                                      Ts: Union[SE3, torch.Tensor], 
                                      n_steps: int,
                                      dt: float,
                                      cutoff_r: float,
                                      max_num_neighbors: int = 100,
                                      eps: float = 0.01,
                                      cluster_method: str = 'knn',
                                      revert_order: bool = False,
                                      voxel_size: Optional[float] = None,
                                      voxel_coord_reduction: Optional[str] = None) -> List[SE3]:
    """_summary_

    Args:
        x (Union[PointCloud, torch.Tensor]): (nX, 3)
        y (Union[PointCloud, torch.Tensor]): (nPose, nY, 3) or (nY, 3)
        Ts (Union[SE3, torch.Tensor]): (nPose, 7)
        n_steps (int): _description_
        dt (float): _description_
        cutoff_r (float): _description_
        max_num_neighbors (int, optional): _description_. Defaults to 100.
        eps (float, optional): _description_. Defaults to 0.01.
        cluster_method (str, optional): _description_. Defaults to 'knn'.

    Returns:
        trajectories: _description_
    """
    if voxel_size is not None:
        if voxel_coord_reduction is None:
            voxel_coord_reduction = 'average'
        x = preprocess.downsample(x, voxel_size=voxel_size, coord_reduction=voxel_coord_reduction)
        y = preprocess.downsample(y, voxel_size=voxel_size, coord_reduction=voxel_coord_reduction)

    x = convert_to_tensor(x)
    y = convert_to_tensor(y)
    trajectories: torch.Tensor = _optimize_pcd_collision_trajectory(x=x, y=y, Ts=convert_to_tensor(Ts), n_steps=n_steps, dt=dt, cutoff_r=cutoff_r, max_num_neighbors=max_num_neighbors, eps=eps, cluster_method=cluster_method, revert_order=revert_order) # (..., nTime, 7)
    if isinstance(Ts, torch.Tensor):
        Ts = SE3(poses=Ts)
    trajectories: List[SE3] = [Ts.new(poses=traj) for traj in trajectories] # List of n_poses * (n_steps, 7) poses
    return trajectories