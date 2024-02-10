from __future__ import annotations
import os

from typing import Union, Optional, List, Tuple, Dict, Any, Iterable, TypeVar, Type, NamedTuple, Sequence
import warnings
from beartype import beartype

import numpy as np

import plotly.graph_objects as go
import matplotlib
import numpy as np

magma_cmap = matplotlib.cm.get_cmap('magma')
viridis_cmap = matplotlib.cm.get_cmap('viridis')

import torch


from .base import DataAbstractBase, Action, Observation
from .se3 import SE3


@torch.jit.script
def _compute_inrange_mask(points: torch.Tensor, bbox: torch.Tensor) -> torch.Tensor:
    assert points.ndim == 2 and points.shape[-1] == 3, f"{points.shape}"
    assert bbox.shape == (3,2), f"{bbox.shape}"

    # Unpack the bounding box
    xmin, xmax = bbox[0,0], bbox[0,1]
    ymin, ymax = bbox[1,0], bbox[1,1]
    zmin, zmax = bbox[2,0], bbox[2,1]

    # Create masks for each dimension
    mask_x = (points[:, 0] >= xmin) & (points[:, 0] <= xmax)
    mask_y = (points[:, 1] >= ymin) & (points[:, 1] <= ymax)
    mask_z = (points[:, 2] >= zmin) & (points[:, 2] <= zmax)

    # Combine masks
    mask = mask_x & mask_y & mask_z
    return mask

@beartype
class PointCloud(Observation):
    data_args_type: Dict[str, type] = {
            'points': torch.Tensor,
            'colors': torch.Tensor,
    }

    metadata_args: List[str] = ['name', 'unit_length']

    points: torch.Tensor
    colors: torch.Tensor
    name: str
    unit_length: str

    @property
    def device(self) -> torch.device:
        return self.points.device
    
    @beartype
    def __init__(self, points: torch.Tensor, 
                 colors: torch.Tensor, 
                 name: str = '',
                 unit_length: str = '1 [m]', 
                 cmap: Optional[str] = None):
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
        self.unit_length = unit_length
    
    def __len__(self) -> int:
        return len(self.points)

    @staticmethod
    def from_numpy(points: np.ndarray, colors: np.ndarray, device: Union[str, torch.device] = 'cpu') -> PointCloud:
        return PointCloud(points=torch.tensor(points, dtype=torch.float32, device=device), colors=torch.tensor(colors, dtype=torch.float32, device=device))

    @staticmethod
    def from_o3d(pcd, device: Union[str, torch.device] = 'cpu') -> PointCloud:
        import open3d as o3d
        assert isinstance(pcd, o3d.geometry.PointCloud)

        points = torch.tensor(np.asarray(pcd.points), dtype=torch.float32, device=device)
        colors = torch.tensor(np.asarray(pcd.colors), dtype=torch.float32, device=device)

        return PointCloud(points=points, colors=colors)

    def to_o3d(self):
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points.detach().cpu())
        pcd.colors = o3d.utility.Vector3dVector(self.colors.detach().cpu())

        return pcd
    
    @staticmethod
    def empty(device: Union[str, torch.device] = 'cpu') -> PointCloud:
        return PointCloud(points=torch.empty((0,3), device=device), colors=torch.empty((0,3), device=device))
    
    @property
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

        from .pcd_utils import transform_points
        
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

    def transformed(self, Ts: Union[torch.Tensor, SE3], squeeze: bool = False) -> Union[PointCloud, List[PointCloud]]:
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
    
    def plotly(self, point_size: float = 5.0, 
               name: Optional[str] = None, 
               opacity: Union[float, torch.Tensor] = 1.0, 
               colors: Optional[torch.Tensor] = None, 
               custom_data: Optional[dict] = None) -> go.Scatter3d:
        return PointCloud.points_to_plotly(pcd=self, point_size=point_size, name=name, opacity=opacity, colors=colors, custom_data=custom_data)

    @staticmethod
    def show_pcd(pcd: Union[PointCloud, torch.Tensor], 
                 point_size: float = 1.0, 
                 name: Optional[str] = None, 
                 opacity: Union[float, torch.Tensor] = 1.0, 
                 colors: Optional[Iterable] = None, 
                 custom_data: Optional[Dict] = None,
                 width = 600,
                 height = 600,
                 bg_color = None
                 ) -> go.Figure:
        data = PointCloud.points_to_plotly(pcd=pcd, point_size=point_size, name=name, opacity=opacity, colors=colors, custom_data=custom_data)

        x_min, x_max = min(data.x), max(data.x)
        y_min, y_max = min(data.y), max(data.y)
        z_min, z_max = min(data.z), max(data.z)

        max_range = max([x_max - x_min, y_max - y_min, z_max - z_min])
        xc, yc, zc = (x_max+x_min)/2, (y_max+y_min)/2, (z_max+z_min)/2
        w = max_range / 2

        fig = go.Figure(data=[data], layout=dict(
            width=width, 
            height=height,
            margin=dict(t=0, r=0, l=0, b=0),
            autosize=False,
            scene=dict(
                aspectmode='cube',
                xaxis=dict(range=[xc-w, xc+w]),
                yaxis=dict(range=[yc-w, yc+w]),
                zaxis=dict(range=[zc-w, zc+w])
            )
        ))
        if bg_color is not None:
            from .visualize import update_background_color
            fig = update_background_color(fig, color=bg_color)
        return fig
    
    def show(self, 
             point_size: float = 1.0, 
             name: Optional[str] = None, 
             opacity: Union[float, torch.Tensor] = 1.0, 
             colors: Optional[torch.Tensor] = None, 
             custom_data: Optional[dict] = None, 
             width = 600, 
             height = 600,
             bg_color = None):
        return PointCloud.show_pcd(pcd=self, point_size=point_size, name=name, opacity=opacity, colors=colors, custom_data=custom_data, width=width, height=height, bg_color=bg_color)
    
    @staticmethod
    def crop_pcd(data: PointCloud, bbox: Union[torch.Tensor, List, Tuple, np.ndarray]):
        if data.is_empty:
            return data           
        points, colors = data.points, data.colors

        bbox = torch.tensor(bbox, dtype=points.dtype, device=points.device)
        assert bbox.shape == (3,2), f"{bbox.shape}"
        in_range_mask = _compute_inrange_mask(points=points, bbox=bbox)

        return data.new(points=points[in_range_mask], colors=colors[in_range_mask])
    
    def crop(self, bbox: Union[torch.Tensor, List, Tuple, np.ndarray]):
        return self.crop_pcd(data=self, bbox=bbox)


        

        
        
