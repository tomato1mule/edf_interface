from typing import Optional, Tuple, List, Union
import torch
from beartype import beartype

from edf_interface import data
from edf_interface.data import transforms
from edf_interface.utils.collision_utils import optimize_pcd_collision_trajectory

@torch.jit.script
def _interpolate_trajectory(init_poses: torch.Tensor,
                            final_poses: torch.Tensor,
                            n_steps: int) -> torch.Tensor:
    assert n_steps >= 1
    assert init_poses.shape == final_poses.shape, f"{init_poses.shape}, {final_poses.shape}"
    assert init_poses.shape[-1] == final_poses.shape[-1], f"{init_poses.shape}, {final_poses.shape}"
    original_shape = init_poses.shape

    init_poses = init_poses.reshape(-1,7) # (n, 7)
    final_poses = final_poses.reshape(-1,7) # (n, 7)
    final_rel_poses = data.se3._multiply(data.se3._inv(init_poses), final_poses) # (n, 7)
    lie = data.se3._log_map(poses = final_rel_poses) # (n, 6): (Rx, Ry, Rz, Vx, Vy, Vz)

    traj = []
    for n in range(n_steps):
        interp = n/n_steps
        rel_poses = data.se3._exp_map(lie*interp)
        abs_poses = data.se3._multiply(init_poses, rel_poses).reshape(original_shape)
        traj.append(abs_poses)
    traj.append(final_poses.reshape(original_shape))
    traj = torch.stack(traj, dim=0) # (n_steps+1, ..., 7)
    return traj.movedim(0,-2) # (..., n_steps+1, 7)

@torch.jit.script
def _extrapolate_trajectory(init_poses: torch.Tensor,
                            reference_poses: torch.Tensor,
                            extrapolate_factor: float,
                            n_steps: int) -> torch.Tensor:
    assert n_steps >= 1
    assert init_poses.shape == reference_poses.shape, f"{init_poses.shape}, {reference_poses.shape}"
    assert init_poses.shape[-1] == reference_poses.shape[-1], f"{init_poses.shape}, {reference_poses.shape}"
    original_shape = init_poses.shape

    init_poses = init_poses.reshape(-1,7) # (n, 7)
    reference_poses = reference_poses.reshape(-1,7) # (n, 7)
    reference_rel_poses = data.se3._multiply(data.se3._inv(init_poses), reference_poses) # (n, 7)
    lie = data.se3._log_map(poses = reference_rel_poses) # (n, 6): (Rx, Ry, Rz, Vx, Vy, Vz)
    lie = extrapolate_factor * lie

    traj = []
    for n in range(n_steps + 1):
        interp = n/n_steps
        rel_poses = data.se3._exp_map(lie*interp)
        abs_poses = data.se3._multiply(init_poses, rel_poses).reshape(original_shape)
        traj.append(abs_poses)
    traj = torch.stack(traj, dim=0) # (n_steps+1, ..., 7)
    return traj.movedim(0,-2) # (..., n_steps+1, 7)

@beartype
def compute_pre_pick_trajectories(pick_poses: data.SE3,
                                  approach_len: float,
                                  n_steps: int) -> List[data.SE3]:
    assert pick_poses.poses.ndim == 2, f"{pick_poses.poses.shape}"
    rel_pose = data.SE3(torch.tensor([1., 0., 0., 0., 0., 0., -approach_len], device=pick_poses.device))
    pre_pick_poses = pick_poses * rel_pose
    trajectories: torch.Tensor = _interpolate_trajectory(init_poses=pre_pick_poses.poses, final_poses=pick_poses.poses, n_steps=n_steps)
    assert trajectories.ndim == 3 and trajectories.shape[:-2] == pick_poses.poses.shape[:-1]
    return [pick_poses.new(poses=traj) for traj in trajectories]

@beartype
def compute_post_pick_trajectories(pick_poses: data.SE3,
                                   lift_len: float, n_steps: int,
                                   include_target_pose: bool = False) -> List[data.SE3]:
    post_pick_poses = data.SE3(pick_poses.poses + torch.tensor([0., 0., 0., 0., 0., 0., lift_len], device=pick_poses.device))
    trajectories = _interpolate_trajectory(init_poses=pick_poses.poses, final_poses=post_pick_poses.poses, n_steps=n_steps)

    return [pick_poses.new(
        poses = traj if include_target_pose else traj[1:]
    ) for traj in trajectories]


@beartype
def compute_pre_place_trajectories(place_poses: data.SE3, 
                                   scene_pcd: data.PointCloud, 
                                   grasp_pcd: data.PointCloud, 
                                   n_steps: int,
                                   dt: float,
                                   cutoff_r: float,
                                   max_num_neighbors: int = 100,
                                   eps: float = 0.01,
                                   cluster_method: str = 'knn',
                                   voxel_size: Optional[float] = None,
                                   voxel_coord_reduction: Optional[str] = None) -> List[data.SE3]:
    trajectories: List[data.SE3] = optimize_pcd_collision_trajectory(x=scene_pcd, 
                                                                     y=grasp_pcd, 
                                                                     Ts=place_poses, 
                                                                     n_steps=n_steps, 
                                                                     dt=dt, 
                                                                     cutoff_r=cutoff_r, 
                                                                     max_num_neighbors=max_num_neighbors, 
                                                                     eps=eps, 
                                                                     cluster_method=cluster_method,
                                                                     revert_order = True,
                                                                     voxel_size=voxel_size,
                                                                     voxel_coord_reduction=voxel_coord_reduction)
    return trajectories

@beartype
def compute_post_place_trajectories(place_poses: data.SE3, 
                                    pre_pick_trajectory: data.SE3, 
                                    n_steps: int,
                                    extrapolate_post_place: Optional[float] = None,
                                    include_target_pose: bool = False) -> List[data.SE3]:
    pre_pick_pose, pick_pose = pre_pick_trajectory[0], pre_pick_trajectory[-1]

    assert len(pick_pose) == len(pre_pick_pose) == 1

    rel_post_place = pick_pose.inv() * pre_pick_pose
    post_place_poses = place_poses * rel_post_place

    if extrapolate_post_place is not None:
        trajectories = _extrapolate_trajectory(
            init_poses=place_poses.poses, 
            reference_poses=post_place_poses.poses,
            n_steps=n_steps,
            extrapolate_factor=extrapolate_post_place,
        )
    else:
        trajectories = _interpolate_trajectory(
            init_poses=place_poses.poses,
            final_poses=post_place_poses.poses,
            n_steps=n_steps,
        )

    return [place_poses.new(
        poses = traj if include_target_pose else traj[1:]
    ) for traj in trajectories]
    
    
def get_inlier_idx_median(x, interval: Union[float, int] = 1.0):
    x=x.detach().cpu()
    std = x.std() * interval
    inlier = (x >= x.median() - std) * (x <= x.median() + std)
    inlier = inlier.nonzero().squeeze(-1)
    return [int(i) for i in inlier.cpu().numpy()]

def get_inlier_idx_mean(x, interval: Union[float, int] = 1.0):
    x=x.detach().cpu()
    std = x.std() * interval
    inlier = (x >= x.mean() - std) * (x <= x.mean() + std)
    inlier = inlier.nonzero().squeeze(-1)
    return [int(i) for i in inlier.cpu().numpy()]

def remove_outliers(xs, critic, interval: Union[float, int] = 1.0):
    inliers_idx = get_inlier_idx_median(x=critic, interval=interval * 1.5)
    xs = [xs[i] for i in inliers_idx]
    critic = torch.stack([critic[i] for i in inliers_idx], dim=0)
    
    inliers_idx = get_inlier_idx_mean(x=critic, interval=interval)
    inliers = [xs[i] for i in inliers_idx]
    critic_inliers = torch.stack([critic[i] for i in inliers_idx], dim=0)
    return inliers, critic_inliers