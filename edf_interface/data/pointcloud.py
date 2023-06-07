from __future__ import annotations
import os

from typing import Union, Optional, List, Tuple, Dict, Any, Iterable, TypeVar, Type, NamedTuple, Sequence
import warnings
from beartype import beartype

import numpy as np
import open3d as o3d

import plotly.graph_objects as go
import matplotlib
import numpy as np

magma_cmap = matplotlib.cm.get_cmap('magma')
viridis_cmap = matplotlib.cm.get_cmap('viridis')

import torch


from .base import DataAbstractBase, Action, Observation
from .se3 import SE3


@beartype
class PointCloud(Observation):
    data_args_hint: Dict[str, type] = {
            'points': torch.Tensor,
            'colors': torch.Tensor,
    }

    metadata_args_hint: Dict[str, type] = {
            'name': str,
    }

    points: torch.Tensor
    colors: torch.Tensor
    name: str

    @property
    def device(self) -> torch.device:
        return self.points.device
    
    @beartype
    def __init__(self, points: torch.Tensor, 
                 colors: torch.Tensor, 
                 name: str = '',
                 device: Optional[Union[str, torch.device]] = None, cmap: Optional[str] = None):
        if device is not None:
            points = points.to(device)
            colors = colors.to(device)
        assert points.device == colors.device


        self.points: torch.Tensor = points # (N,3)
        self.colors: torch.Tensor = colors # (N,3)
        if cmap is None:
            if self.colors.shape != self.points.shape:
                raise ValueError(f"shape of the color ({self.colors.shape}) does not match with ({self.points.shape}). Use cmap argument if using scalar features")
        elif cmap == 'viridis':
            assert (self.colors.ndim + 1 == self.points.ndim and self.colors.shape == self.points.shape[:1])\
                    or (self.colors.shape[-1] == 1 and self.colors.shape[:1] == self.points.shape[:1])
            colors = viridis_cmap(self.colors)[..., :3] # https://plotly.com/python/v3/matplotlib-colorscales/
            self.colors: torch.Tensor = torch.tensor(colors, device=self.colors.device, dtype=self.colors.dtype)
        elif cmap == 'magma':
            assert (self.colors.ndim + 1 == self.points.ndim and self.colors.shape == self.points.shape[:1])\
                    or (self.colors.shape[-1] == 1 and self.colors.shape[:1] == self.points.shape[:1])
            colors = magma_cmap(self.colors)[..., :3] # https://plotly.com/python/v3/matplotlib-colorscales/
            self.colors: torch.Tensor = torch.tensor(colors, device=self.colors.device, dtype=self.colors.dtype)
        else:
            raise ValueError(f"Unknown cmap: {cmap}")
        
        self.name = name
    
    def __len__(self) -> int:
        return len(self.points)

    @staticmethod
    def from_numpy(points: np.ndarray, colors: np.ndarray, device: Union[str, torch.device] = 'cpu') -> PointCloud:
        return PointCloud(points=torch.tensor(points, dtype=torch.float32, device=device), colors=torch.tensor(colors, dtype=torch.float32, device=device), device=device)

    @staticmethod
    def from_o3d(pcd: o3d.geometry.PointCloud, device: Union[str, torch.device] = 'cpu') -> PointCloud:
        points = torch.tensor(np.asarray(pcd.points), dtype=torch.float32, device=device)
        colors = torch.tensor(np.asarray(pcd.colors), dtype=torch.float32, device=device)

        return PointCloud(points=points, colors=colors, device=device)

    def to_o3d(self) -> o3d.geometry.PointCloud:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points.detach().cpu())
        pcd.colors = o3d.utility.Vector3dVector(self.colors.detach().cpu())

        return pcd
    
    @staticmethod
    def empty(device: Union[str, torch.device] = 'cpu') -> PointCloud:
        return PointCloud(points=torch.empty((0,3), device=device), colors=torch.empty((0,3), device=device))
    
    def is_empty(self) -> bool:
        if len(self.points) == 0:
            return True
        else:
            return False
    
    @staticmethod
    def transform_pcd(pcd, Ts: Union[torch.Tensor, SE3], squeeze: bool = False) -> Union[List[PointCloud], PointCloud]:
        assert isinstance(pcd, PointCloud)
        if isinstance(Ts, SE3):
            Ts = Ts.poses.detach().clone()
        ndim = Ts.ndim
        assert ndim <= 2 and Ts.shape[-1] == 7
        assert pcd.device == Ts.device

        from diffusion_edf.pc_utils import transform_points
        
        points = transform_points(points = pcd.points, Ts = Ts)
        if points.ndim == 3:
            output: List[PointCloud] = [PointCloud(points=point, colors=pcd.colors) for point in points]
        else:
            output: List[PointCloud] = [PointCloud(points=points, colors=pcd.colors)]
        if squeeze:
            assert len(output) == 1
            return output[0]
        else:
            return output

    def transformed(self, Ts: Union[torch.Tensor, SE3], squeeze: bool = False) -> PointCloud:
        return PointCloud.transform_pcd(pcd=self, Ts=Ts, squeeze=squeeze)
    
    @staticmethod
    def merge(*args) -> PointCloud:
        points = []
        colors = []
        for pcd in args:
            points.append(pcd.points)
            colors.append(pcd.colors)
        points = torch.cat(points, dim=-2)
        colors = torch.cat(colors, dim=-2)

        return PointCloud(points=points, colors=colors)

    @staticmethod
    def points_to_plotly(pcd: Union[PointCloud, torch.Tensor], 
                         point_size: float = 1.0, 
                         name: Optional[str] = None, 
                         opacity: Union[float, torch.Tensor] = 1.0, 
                         colors: Optional[Iterable] = None, 
                         custom_data: Optional[Dict] = None) -> go.Scatter3d:
        if colors is not None:
            colors = torch.tensor(colors)
        if isinstance(pcd, PointCloud):
            points: torch.Tensor = pcd.points
            if colors is None:
                colors: torch.Tensor = pcd.colors
        if isinstance(pcd, torch.Tensor):
            assert pcd.ndim==2 and pcd.shape[-1] == 3, f"pcd must be 3-dimensional pointcloud, but pcd with shape {pcd.shape} is given."
            points: torch.Tensor = pcd
            if colors is None:
                colors: torch.Tensor = torch.zeros(len(points), 3)
        colors = colors.detach().cpu()
        if colors.ndim==1 and len(colors) == 3:
            colors = colors.expand(len(pcd), 3)

        pcd_marker = {}

        if isinstance(opacity, torch.Tensor):
            assert len(opacity) == pcd.__len__()
            colors = torch.cat([colors, opacity.detach().cpu().unsqueeze(-1)], dim=-1)
        elif type(opacity) == float:
            pcd_marker['opacity'] = opacity

        pcd_marker['size'] = point_size
        pcd_marker['color'] = colors


        plotly_kwargs = dict(x=points[:,0].detach().cpu(), y=points[:,1].detach().cpu(), z=points[:,2].detach().cpu(), mode='markers', marker=pcd_marker)
        if name is not None:
            plotly_kwargs['name'] = name

        if custom_data is not None:
            _customdata = []
            hover_template = ''
            for i,(k,v) in enumerate(custom_data.items()):
                _customdata.append(v)
                hover_template += f'<b>{k}</b>: ' + '%{customdata' +f'[{i}]' + ':,.2f}<br>'
            hover_template = hover_template.lstrip('<br>')
            hover_template += '<extra></extra>'
            plotly_kwargs['hovertemplate'] = hover_template
            print(hover_template)
            plotly_kwargs['customdata'] = torch.stack(_customdata, dim=-2)

        return go.Scatter3d(**plotly_kwargs)
    
    @staticmethod
    def show_pcd(pcd: Union[PointCloud, torch.Tensor], 
                 point_size: float = 1.0, 
                 name: Optional[str] = None, 
                 opacity: Union[float, torch.Tensor] = 1.0, 
                 colors: Optional[Iterable] = None, 
                 custom_data: Optional[Dict] = None,
                 width = 1600,
                 height = 1200,
                 ) -> go.Figure:
        
        data = PointCloud.points_to_plotly(pcd=pcd, point_size=point_size, name=name, opacity=opacity, colors=colors, custom_data=custom_data)
        fig = go.Figure(data=[data], layout=dict(width=width, height=height))
        return fig
    
    def plotly(self, point_size: float = 5.0, name: Optional[str] = None, opacity: Union[float, torch.Tensor] = 1.0, colors: Optional[torch.Tensor] = None, custom_data: Optional[dict] = None) -> go.Scatter3d:
        return PointCloud.points_to_plotly(pcd=self, point_size=point_size, name=name, opacity=opacity, colors=colors, custom_data=custom_data)
    
    def show(self, point_size: float = 5.0, name: Optional[str] = None, opacity: Union[float, torch.Tensor] = 1.0, colors: Optional[torch.Tensor] = None, custom_data: Optional[dict] = None, width = 700, height=700):
        return PointCloud.show_pcd(pcd=self, point_size=point_size, name=name, opacity=opacity, colors=colors, custom_data=custom_data, width=width, height=height)
    


        

        
        
