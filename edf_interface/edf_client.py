from typing import Optional
import pickle

from beartype import beartype
import torch

from edf_interface.pyro import get_service_proxy
from edf_interface.data import SE3, PointCloud, TargetPoseDemo, DemoSequence

@beartype
class EdfClient():
    def __init__(self, env_server_name: str = 'env',
                 agent_sever_name: str = 'agent'):
        self.env_service = get_service_proxy(env_server_name)
        self.agent_service = get_service_proxy(agent_sever_name)

    def get_current_poses(self, **kwargs) -> SE3:
        data_dict = self.env_service.get_current_poses(**kwargs)
        return SE3.from_data_dict(data_dict=data_dict)
    
    def observe_scene(self, **kwargs) -> PointCloud:
        data_dict = self.env_service.observe_scene(**kwargs)
        return PointCloud.from_data_dict(data_dict=data_dict)
    
    def observe_grasp(self, **kwargs) -> PointCloud:
        data_dict = self.env_service.observe_grasp(**kwargs)
        return PointCloud.from_data_dict(data_dict=data_dict)
    
    def move_se3(self, target_poses: SE3, **kwargs) -> bool:
        target_poses = target_poses.get_data_dict(serialize=True)
        success = self.env_service.move_se3(target_poses=target_poses, **kwargs)
        return success
    
    def infer_target_poses(self, scene_pcd: PointCloud, 
                           task_name: str,
                           grasp_pcd: Optional[PointCloud] = None,
                           current_poses: Optional[SE3] = None, 
                           **kwargs) -> SE3:
        scene_pcd = scene_pcd.get_data_dict(serialize=True)
        if grasp_pcd is not None:
            grasp_pcd = grasp_pcd.get_data_dict(serialize=True)
        else:
            grasp_pcd = {}
        if current_poses is not None:
            current_poses = current_poses.get_data_dict(serialize=True)
        else:
            current_poses = {}
        target_poses_dict = self.agent_service.infer_target_poses(scene_pcd=scene_pcd, grasp_pcd=grasp_pcd, task_name=task_name, current_poses=current_poses, **kwargs)
        return SE3.from_data_dict(target_poses_dict)
