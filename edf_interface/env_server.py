from abc import ABCMeta, abstractmethod
from typing import Optional, List, Dict

from beartype import beartype

import Pyro5.api, Pyro5.server
from edf_interface.pyro import PyroServer
from edf_interface.data import DemoDataset, TargetPoseDemo, DemoSequence, SE3, PointCloud, DataAbstractBase, serialize_if_data


@beartype
class EnvHandleAbstractBase(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def get_current_poses(self) -> SE3:
        pass

    @abstractmethod
    def observe_scene(self) -> PointCloud:
        pass

    @abstractmethod
    def observe_grasp(self) -> PointCloud:
        pass

    @abstractmethod
    def move_se3(self, target_poses: SE3) -> bool:
        pass

    # @abstractmethod
    # def plan(self, target_poses: SE3):
    #     pass

    # @abstractmethod
    # def execute(self, plan) -> SE3:
    #     pass

    # @abstractmethod
    # def control_gripper(self, gripper_val: float):
    #     pass


@beartype
class EnvService:
    env_handle: EnvHandleAbstractBase

    def __init__(self, env_handle: EnvHandleAbstractBase) -> None:
        self.env_handle = env_handle
        
    @Pyro5.api.expose
    @serialize_if_data
    def get_current_poses(self, *args, **kwargs) -> SE3:
        return self.env_handle.get_current_poses(*args, **kwargs)
    
    @Pyro5.api.expose
    @serialize_if_data
    def observe_scene(self, *args, **kwargs) -> PointCloud:
        return self.env_handle.observe_scene(*args, **kwargs)
    
    @Pyro5.api.expose
    @serialize_if_data
    def observe_grasp(self, *args, **kwargs) -> PointCloud:
        return self.env_handle.observe_grasp(*args, **kwargs)
    
    @Pyro5.api.expose
    @serialize_if_data
    def move_se3(self, target_poses: Dict, *args, **kwargs) -> bool:
        target_poses = SE3.from_data_dict(target_poses)
        return self.env_handle.move_se3(target_poses=target_poses, *args, **kwargs)
    
@beartype
class EnvServer:
    env_service: EnvService
    pyro_server: PyroServer
    server_name: str

    def __init__(self, env_handle: EnvHandleAbstractBase,
                 server_name: str = 'env',
                 init_nameserver: Optional[bool] = None):
        self.server_name = server_name
        self.env_service = EnvService(env_handle=env_handle)
        self.pyro_server = PyroServer(self.env_service, 
                                      server_name=self.server_name,
                                      init_nameserver=init_nameserver)
        
    def run(self, nonblocking: bool = False):
        return self.pyro_server.run(nonblocking=nonblocking)
    
    def close(self):
        self.pyro_server.close()