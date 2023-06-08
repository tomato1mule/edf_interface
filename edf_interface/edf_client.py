import pickle

from beartype import beartype
import torch

from edf_interface.pyro import get_service_proxy
from edf_interface.data import SE3, PointCloud, TargetPoseDemo, DemoSequence
from edf_interface.env_server import EnvService

@beartype
class EdfClient():
    env_service: EnvService
    def __init__(self, env_server_name: str = 'env'):
        self.env_service = get_service_proxy(env_server_name)

    def get_current_poses(self) -> SE3:
        data_dict = self.env_service.get_current_poses()
        return SE3.from_data_dict(data_dict=data_dict)
    
    def observe_scene(self) -> PointCloud:
        data_dict = self.env_service.observe_scene()
        return PointCloud.from_data_dict(data_dict=data_dict)
    
    def observe_grasp(self) -> PointCloud:
        data_dict = self.env_service.observe_grasp()
        return PointCloud.from_data_dict(data_dict=data_dict)
    
    def move_se3(self, target_poses: SE3) -> bool:
        target_poses = target_poses.get_data_dict(serialize=True)
        success = self.env_service.move_se3(target_poses=target_poses)
        return success
    
