# edf_interface
EDF interface

# Installation
Execute the following command in your terminal.
```shell
pip install -e .
```
Install PyTorch if you don't have it.
```shell
pip install torch==1.13.1
```

# Usage
Use @expose decorator to share server's class methods with clients.

## Environment Server Example
```python
from edf_interface.data import SE3, PointCloud
from edf_interface.pyro import PyroServer, expose

class EnvService():
    def __init__(self): ...

    @expose
    def get_current_poses(self) -> SE3: ...

    @expose
    def observe_scene(self) -> PointCloud: ...

    @expose
    def observe_grasp(self) -> PointCloud: ...

    @expose
    def move_se3(self, target_poses: SE3) -> bool: ...

service = EnvService()
server = PyroServer(server_name='env', init_nameserver=True)
server.register_service(service=service)
server.run(nonblocking=False) # set nonblocking = True if you want to run server in another thread.

server.close()
```

## Agent Server Example
```python
from edf_interface.data import SE3, PointCloud
from edf_interface.pyro import PyroServer, expose

class AgentService():
    def __init__(self):
        pass

    @expose
    def infer_target_poses(self, scene_pcd: PointCloud, 
                           task_name: str,
                           grasp_pcd: PointCloud,
                           current_poses: SE3) -> SE3: ...

service = AgentService()
server = PyroServer(server_name='agent', init_nameserver=False)
server.register_service(service=service)
server.run(nonblocking=False) # set nonblocking = True if you want to run server in another thread.

server.close()
```

## Client Example
```python
from edf_interface.data import SE3, PointCloud
from edf_interface.pyro import get_service_proxy, wrap_remote

class EdfClient():
    def __init__(self, env_server_name: str = 'env',
                 agent_sever_name: str = 'agent'):
        self._env_service = get_service_proxy(env_server_name)
        self._agent_service = get_service_proxy(agent_sever_name)
        self._register_remote_methods(self._env_service)
        self._register_remote_methods(self._agent_service)

    def _register_remote_methods(self, service):
        service._pyroBind()
        for method in service._pyroMethods:
            if hasattr(self, method):
                setattr(self, method, wrap_remote(getattr(service, method)))

    def get_current_poses(self) -> SE3: ...
    
    def observe_scene(self) -> PointCloud: ...
    
    def observe_grasp(self) -> PointCloud: ...

    def move_se3(self, target_poses: SE3) -> bool: ...

    def infer_target_poses(self, scene_pcd: PointCloud, 
                           task_name: str,
                           grasp_pcd: Optional[PointCloud] = None,
                           current_poses: Optional[SE3] = None, 
                           **kwargs) -> SE3: ...

client = EdfClient(env_server_name='env', agent_sever_name='agent')
```