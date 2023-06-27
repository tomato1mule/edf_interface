from typing import Optional, Tuple
import torch
from beartype import beartype

from edf_interface import data
from edf_interface.data import transforms
from edf_interface.utils.collision_utils import optimize_pcd_collision

# def compute_pre_post_pick(scene: data.PointCloud, grasp: data.PointCloud, pick_poses: data.SE3) -> Tuple[data.SE3, data.SE3]:
#     _, pre_pick_poses = optimize_pcd_collision(x=scene, y=grasp, 
#                                                 cutoff_r = 0.03, dt=0.01, eps=1., iters=50,
#                                                 rel_pose=pick_poses)
#     post_pick_poses = pre_pick_poses

#     return pre_pick_poses, post_pick_poses

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