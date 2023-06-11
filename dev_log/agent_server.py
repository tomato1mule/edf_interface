from abc import ABCMeta, abstractmethod
from typing import Optional, List, Dict

from beartype import beartype

import Pyro5.api, Pyro5.server
from edf_interface.pyro import PyroServer
from edf_interface.data import DemoDataset, TargetPoseDemo, DemoSequence, SE3, PointCloud, DataAbstractBase, serialize_if_data


@beartype
class AgentHandleAbstractBase(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def infer_target_poses(self, scene_pcd: PointCloud, 
                           task_name: str,
                           grasp_pcd: PointCloud,
                           current_poses: SE3) -> SE3:
        pass


@beartype
class AgentService:
    agent_handle: AgentHandleAbstractBase

    def __init__(self, agent_handle: AgentHandleAbstractBase) -> None:
        self.agent_handle = agent_handle
        
    @Pyro5.api.expose
    @serialize_if_data
    def infer_target_poses(self, scene_pcd: Dict, 
                           task_name:str, 
                           grasp_pcd: Dict, 
                           current_poses: Dict,
                           **kwargs,
                           ):
        scene_pcd = PointCloud.from_data_dict(scene_pcd)
        if grasp_pcd:
            grasp_pcd = PointCloud.from_data_dict(grasp_pcd)
        else:
            grasp_pcd = None
        if current_poses:
            current_poses = SE3.from_data_dict(current_poses)
        else:
            current_poses = None

        return self.agent_handle.infer_target_poses(scene_pcd=scene_pcd, grasp_pcd=grasp_pcd, task_name=task_name, current_poses=current_poses, **kwargs)
    
@beartype
class AgentServer:
    agent_service: AgentService
    pyro_server: PyroServer
    server_name: str

    def __init__(self, agent_handle: AgentHandleAbstractBase,
                 server_name: str = 'agent',
                 init_nameserver: Optional[bool] = None):
        self.server_name = server_name
        self.agent_service = AgentService(agent_handle=agent_handle)
        self.pyro_server = PyroServer(self.agent_service, 
                                      server_name=self.server_name,
                                      init_nameserver=init_nameserver)
        
    def run(self, nonblocking: bool = False):
        return self.pyro_server.run(nonblocking=nonblocking)
    
    def close(self):
        self.pyro_server.close()