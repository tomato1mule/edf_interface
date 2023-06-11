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

# Example
Please run the following notebooks in order with Jupyter:
1. 'env_server.ipynb'
2. 'agent_server.ipynb'
3. 'client.ipynb'

# Usage
Use @expose decorator to share server's class methods with clients.

## Environment Server Example
```python
from edf_interface.data import SE3, PointCloud
from edf_interface.pyro import PyroServer, expose

class EnvService():
    def __init__(self): ...

    @expose
    def get_current_poses(self) -> SE3: 
        <YOUR CODE HERE>

    @expose
    def observe_scene(self) -> PointCloud: 
        <YOUR CODE HERE>

    @expose
    def observe_grasp(self) -> PointCloud: 
        <YOUR CODE HERE>

    @expose
    def move_se3(self, target_poses: SE3) -> bool: 
        <YOUR CODE HERE>

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
                           current_poses: SE3) -> SE3: 
        <YOUR CODE HERE>

service = AgentService()
server = PyroServer(server_name='agent', init_nameserver=False)
server.register_service(service=service)
server.run(nonblocking=False) # set nonblocking = True if you want to run server in another thread.

server.close()
```

## Client Example
Methods are only for type hinting. You do not have to write the codes.
```python
from edf_interface.data import SE3, PointCloud
from edf_interface.pyro import PyroClientBase

class ExampleClient(PyroClientBase):
    def __init__(self, env_server_name: str = 'env',
                 agent_sever_name: str = 'agent'):
        super().__init__(service_names=[env_server_name, agent_sever_name])

    def get_current_poses(self, **kwargs) -> SE3: ... 
    
    def observe_scene(self, **kwargs) -> PointCloud: ...
    
    def observe_grasp(self, **kwargs) -> PointCloud: ...

    def move_se3(self, target_poses: SE3, **kwargs) -> bool: ...

    def infer_target_poses(self, scene_pcd: PointCloud, 
                           task_name: str,
                           grasp_pcd: Optional[PointCloud] = None,
                           current_poses: Optional[SE3] = None, 
                           **kwargs) -> SE3: ...

client = ExampleClient(env_server_name='env', agent_sever_name='agent')
```