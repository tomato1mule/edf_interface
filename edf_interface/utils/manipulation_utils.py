from typing import Optional, Tuple, List, Union
import torch
from beartype import beartype

from edf_interface import data
from edf_interface.data import transforms
from edf_interface.utils.collision_utils import optimize_pcd_collision, _optimize_pcd_collision_once

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
    traj = torch.stack(traj, dim=0) # (n_steps, ..., 7)
    return traj.movedim(0,-2) # (..., n_steps, 7)

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
def compute_post_pick(pick_poses: data.SE3,
                      lift_len: float) -> data.SE3:
    post_pick_poses = data.SE3(pick_poses.poses + torch.tensor([0., 0., 0., 0., 0., 0., lift_len], device=pick_poses.device))

    return post_pick_poses


# @beartype
# def compute_pre_place(scene: data.PointCloud, 
#                       grasp: data.PointCloud, 
#                       place_poses: data.SE3, 
#                       n_steps: int,
#                       cutoff_r: float,
#                       dt: float = 0.01,
#                       eps: float = 1.,):
#     grasp = grasp.transformed(Ts=grasp_pose)


#     if rel_pose is not None:
#         if isinstance(y, PointCloud):
#             y = y.transformed(Ts=rel_pose)[0]
#         else:
#             raise NotImplementedError

#     Ts = []
#     for _ in range(iters):
#         y, T, done = _optimize_pcd_collision_once(x=x, y=y, cutoff_r=cutoff_r, dt=dt, eps=eps)
#         Ts.append(T)
#         if done:
#             break

#     if rel_pose is not None:
#         Ts.append(rel_pose)
#     return y, SE3.multiply(*Ts)


#     _, pre_place_poses = optimize_pcd_collision(x=scene, y=grasp, 
#                                                 cutoff_r = cutoff_r, dt=dt, eps=eps, iters=n_steps,
#                                                 rel_pose=place_poses)
    

#     return pre_place_poses






##### Deprecated #####

@beartype
def compute_pre_post_pick(pick_poses: data.SE3,
                          approach_len: float,
                          lift_len: float) -> Tuple[data.SE3, data.SE3]:
    pre_pick_poses = pick_poses * data.SE3(torch.tensor([1., 0., 0., 0., 0., 0., -approach_len], device=pick_poses.device))
    post_pick_poses = data.SE3(pick_poses.poses + torch.tensor([0., 0., 0., 0., 0., 0., lift_len], device=pick_poses.device))

    return pre_pick_poses, post_pick_poses

@beartype
def compute_pre_post_place(scene: data.PointCloud, 
                           grasp: data.PointCloud, 
                           place_poses: data.SE3, 
                           pre_pick_pose: data.SE3, 
                           pick_pose: data.SE3, 
                           cutoff_r: float,
                           dt: float = 0.01,
                           eps: float = 1.,
                           iters: int = 5,
                           extrapolate_post_place: Optional[float] = None) -> Tuple[data.SE3, data.SE3]:
    assert len(pick_pose) == len(pre_pick_pose) == 1

    _, pre_place_poses = optimize_pcd_collision(x=scene, y=grasp, 
                                                cutoff_r = cutoff_r, dt=dt, eps=eps, iters=iters,
                                                rel_pose=place_poses)
    rel_post_place = pick_pose.inv() * pre_pick_pose
    if extrapolate_post_place is not None:
        R = transforms.quaternion_to_matrix(rel_post_place.poses[..., :4])
        x = rel_post_place.poses[..., 4:].unsqueeze(-1)
        T = torch.cat([R,x],dim=-1)
        T = torch.cat([T, torch.cat([torch.zeros_like(T[...,:1,:3]), torch.ones_like(T[...,:1,3:])], dim=-1)], dim=-2)
        lie = transforms.se3_log_map(T)
        T = transforms.se3_exp_map(lie*extrapolate_post_place)

        q = transforms.matrix_to_quaternion(T[...,:3, :3])
        x = T[...,:3,-1]
        T = torch.cat([q,x,], dim=-1)
        rel_post_place = data.SE3(poses=T)

    post_place_poses = place_poses * rel_post_place

    return pre_place_poses, post_place_poses