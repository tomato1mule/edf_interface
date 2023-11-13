from typing import Union, Optional, List, Tuple, Dict, Any, Callable, Sequence
from functools import partial
import random

from beartype import beartype
import torch
import numpy as np

from edf_interface.data import DemoDataset, TargetPoseDemo, DemoSequence, SE3, PointCloud, DataAbstractBase
from edf_interface.data.utils import units_to_str, str_to_units
from edf_interface.data.pcd_utils import voxel_filter
from edf_interface.data import transforms


class PreprocessDataTypeException(Exception):
    pass

class PreprocessNonDataException(Exception):
    pass

@beartype
def compose_procs(proc_fns: List[Callable]) -> Callable:
    def composed(x):
        for proc in proc_fns:
            x = proc(x)
        return x
    return composed

@beartype
def recursive_apply(fn):
    @beartype
    def _recurse(data: Union[DataAbstractBase, Any], *args, **kwargs) -> Union[DataAbstractBase, Any]:
        try:
            data = fn(data=data, *args, **kwargs)
            return data
        except PreprocessDataTypeException:
            data_args = data.data_args_type.keys()
            data_kwargs = {}
            for arg in data_args:
                obj = getattr(data, arg)
                obj = _recurse(data=obj, *args, **kwargs)
                data_kwargs[arg] = obj
            return data.new(**data_kwargs)
        except PreprocessNonDataException:
            return data

    return _recurse


@beartype
@recursive_apply
def rescale(data: Union[DataAbstractBase, Any], rescale_factor: float) -> DataAbstractBase:
    if type(data) == PointCloud:
        val, unit = str_to_units(data.unit_length)
        val=val/rescale_factor
        return data.new(points = data.points * rescale_factor,
                        colors = data.colors * 1.,
                        unit_length = units_to_str(val=val, unit=unit))
    elif type(data) == SE3:
        val, unit = str_to_units(data.unit_length)
        val=val/rescale_factor
        poses = data.poses * torch.tensor([1., 1., 1., 1., rescale_factor, rescale_factor, rescale_factor], dtype=data.poses.dtype, device=data.poses.device).unsqueeze(-2)
        return data.new(poses = poses,
                        unit_length = units_to_str(val=val, unit=unit))
    else:
        if isinstance(data, DataAbstractBase):
            raise PreprocessDataTypeException(f"Unsupported data type: {type(data)}")
        else:
            raise PreprocessNonDataException(f"Unsupported primitive type: {type(data)}")

@beartype
@recursive_apply
def downsample(data: Union[DataAbstractBase, Any], voxel_size: float, coord_reduction: str = "average") -> DataAbstractBase:
    if type(data) == PointCloud:
        if data.is_empty:
            return data
        else:
            points, colors = voxel_filter(points=data.points, features=data.colors, voxel_size=voxel_size, coord_reduction=coord_reduction)
            return data.new(points=points, colors=colors)
    else:
        if isinstance(data, DataAbstractBase):
            raise PreprocessDataTypeException(f"Unsupported data type: {type(data)}")
        else:
            raise PreprocessNonDataException(f"Unsupported primitive type: {type(data)}")

@beartype
@recursive_apply
def change_frame(data: Union[DataAbstractBase, Any], frame: Union[torch.Tensor, SE3]) -> DataAbstractBase: 
    if type(data) == PointCloud:
        if data.is_empty:
            return data
        else:
            if isinstance(frame, SE3):
                frame: torch.Tensor = frame.poses
            assert frame.shape == (7,) or frame.shape == (1,7), f"frame.shape must be (1, 7) or (7,) but {frame.shape} is given."
            q, x = frame[..., :4], frame[..., 4:]
            assert data.points.ndim == 2 and data.points.shape[-1] == 3, f"{data.points.shape}"
            points = transforms.quaternion_apply(q, data.points) + x
            return data.new(points=points)
    elif type(data) == SE3:
        if data.is_empty:
            return data
        else:
            if isinstance(frame, SE3):
                frame: torch.Tensor = frame.poses
            assert frame.shape == (7,) or frame.shape == (1,7), f"frame.shape must be (1, 7) or (7,) but {frame.shape} is given."
            q, x = frame[..., :4], frame[..., 4:]
            assert data.poses.ndim == 2 and data.poses.shape[-1] == 7, f"{data.poses.shape}"
            q_new = transforms.quaternion_multiply(q, data.poses[..., :4])
            q_new = transforms.normalize_quaternion(q_new)
            x_new = transforms.quaternion_apply(q, data.poses[...,4:]) + x
            return data.new(poses = torch.cat([q_new, x_new], dim=-1))
    elif type(data) == TargetPoseDemo:
        scene_pcd = change_frame(data=data.scene_pcd, frame=frame)
        target_poses = change_frame(data=data.target_poses, frame=frame)
        return data.new(scene_pcd=scene_pcd, target_poses=target_poses)
    else:
        if isinstance(data, DataAbstractBase):
            raise PreprocessDataTypeException(f"Unsupported data type: {type(data)}")
        else:
            raise PreprocessNonDataException(f"Unsupported primitive type: {type(data)}")


@torch.jit.script
def compute_inrange_mask(points: torch.Tensor, bbox: torch.Tensor) -> torch.Tensor:
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
def _crop_bbox_pcd(data: PointCloud, bbox: Union[torch.Tensor, List, Tuple, np.ndarray], targets: Optional[List[str]] = None):
    if data.is_empty:
        return data
    elif targets is not None:
        if data.name not in targets:
            return data    
        
    points, colors = data.points, data.colors

    bbox = torch.tensor(bbox, dtype=points.dtype, device=points.device)
    assert bbox.shape == (3,2), f"{bbox.shape}"
    in_range_mask = compute_inrange_mask(points=points, bbox=bbox)

    return data.new(points=points[in_range_mask], colors=colors[in_range_mask])
    

# @beartype
# def _crop_bbox_pose_demo(data: TargetPoseDemo, bbox: Union[torch.Tensor, List, Tuple, np.ndarray], targets: Optional[List[str]] = None) -> TargetPoseDemo:
#     return data.new(scene_pcd=_crop_bbox_pcd(data=data.scene_pcd, bbox=bbox))

@beartype
@recursive_apply
def crop_bbox(data: Union[DataAbstractBase, Any], bbox: Union[torch.Tensor, List, Tuple, np.ndarray], targets: Optional[List[str]] = None):
    if type(data) == PointCloud:
        return _crop_bbox_pcd(data=data, bbox=bbox, targets=targets)
    # if type(data) == TargetPoseDemo:
    #     return _crop_bbox_pose_demo(data=data, bbox=bbox, targets=targets)
    else:
        if isinstance(data, DataAbstractBase):
            raise PreprocessDataTypeException(f"Unsupported data type: {type(data)}")
        else:
            raise PreprocessNonDataException(f"Unsupported primitive type: {type(data)}")
        
        

@beartype
@recursive_apply
def color_jitter(data: Union[DataAbstractBase, Any], std: float, prob: float = 1.0) -> DataAbstractBase:
    if type(data) == PointCloud:
        if data.is_empty or prob < random.random():
            return data
        else:
            colors = data.colors + torch.rand_like(data.colors)*std
            return data.new(colors=colors)
    else:
        if isinstance(data, DataAbstractBase):
            raise PreprocessDataTypeException(f"Unsupported data type: {type(data)}")
        else:
            raise PreprocessNonDataException(f"Unsupported primitive type: {type(data)}")
        
        
@beartype
@recursive_apply
def pos_jitter(data: Union[DataAbstractBase, Any], std: float, prob: float = 1.0) -> DataAbstractBase:
    if type(data) == PointCloud:
        if data.is_empty or prob < random.random():
            return data
        else:
            points = data.points + torch.rand_like(data.points)*std
            return data.new(points=points)
    else:
        if isinstance(data, DataAbstractBase):
            raise PreprocessDataTypeException(f"Unsupported data type: {type(data)}")
        else:
            raise PreprocessNonDataException(f"Unsupported primitive type: {type(data)}")




def _rgb_to_hsl(rgb):
    """
    Convert an RGB image to HSL.
    :param rgb: A PyTorch tensor of shape (N, 3) representing RGB values.
    :return: A PyTorch tensor of shape (N, 3) representing HSL values.
    """
    # Normalize RGB values to the range 0-1
    rgb = rgb.float()

    # Calculate min and max values for each pixel
    min_val, _ = torch.min(rgb, dim=1)
    max_val, _ = torch.max(rgb, dim=1)

    # Calculate Luminance
    l = (max_val + min_val) / 2

    # Initialize Saturation and Hue
    s = torch.zeros_like(l)
    h = torch.zeros_like(l)

    # Calculate Saturation
    delta = max_val - min_val
    non_zero_delta = delta != 0

    # Saturation for non-zero delta
    s[non_zero_delta] = delta[non_zero_delta] / (1 - torch.abs(2 * l[non_zero_delta] - 1))

    # Calculate Hue
    # Red is max
    red_max = (max_val == rgb[:, 0]) & non_zero_delta
    h[red_max] = ((rgb[red_max, 1] - rgb[red_max, 2]) / delta[red_max]) % 6

    # Green is max
    green_max = (max_val == rgb[:, 1]) & non_zero_delta
    h[green_max] = ((rgb[green_max, 2] - rgb[green_max, 0]) / delta[green_max]) + 2

    # Blue is max
    blue_max = (max_val == rgb[:, 2]) & non_zero_delta
    h[blue_max] = ((rgb[blue_max, 0] - rgb[blue_max, 1]) / delta[blue_max]) + 4

    # Normalize Hue to range [0, 1]
    h = h / 6

    return torch.stack([h, s, l], dim=1)


def _hsl_to_rgb(hsl):
    """
    Convert an HSL image to RGB.
    :param hsl: A PyTorch tensor of shape (N, 3) representing HSL values.
    :return: A PyTorch tensor of shape (N, 3) representing RGB values.
    """

    def hue_to_rgb(p, q, t):
        t = torch.where(t < 0, t + 1, t)
        t = torch.where(t > 1, t - 1, t)
        return torch.where(t < 1/6, p + (q - p) * 6 * t,
               torch.where(t < 1/2, q,
               torch.where(t < 2/3, p + (q - p) * (2/3 - t) * 6, p)))

    h, s, l = hsl[:, 0], hsl[:, 1], hsl[:, 2]

    q = torch.where(l < 0.5, l * (1 + s), l + s - l * s)
    p = 2 * l - q

    r = hue_to_rgb(p, q, h + 1/3)
    g = hue_to_rgb(p, q, h)
    b = hue_to_rgb(p, q, h - 1/3)

    return torch.stack([r, g, b], dim=1)



@beartype
@recursive_apply
def randomize_hsl(data: Union[DataAbstractBase, Any], hrange: float, srange: float, lrange: float, prob: float = 1.0) -> DataAbstractBase:
    if type(data) == PointCloud:
        if data.is_empty or prob < random.random():
            return data
        else:
            assert data.colors.ndim == 2 and data.colors.shape[-1] == 3, f"{data.colors.shape}" # (N,3)
            hsl = _rgb_to_hsl(data.colors)
            assert hsl.ndim == 2 and hsl.shape[-1] == 3, f"{hsl.shape}" # (N,3)
            hsl = hsl + ((torch.rand_like(hsl[0,:])-0.5) * torch.tensor([hrange, srange, lrange], device=hsl.device, dtype=hsl.dtype))
            hsl = torch.clip(hsl, min=torch.tensor([-100., 0., 0.], device=hsl.device, dtype=hsl.dtype), max=torch.tensor([100., 1., 1.], device=hsl.device, dtype=hsl.dtype))
            colors = _hsl_to_rgb(hsl)
            return data.new(colors=colors)
    else:
        if isinstance(data, DataAbstractBase):
            raise PreprocessDataTypeException(f"Unsupported data type: {type(data)}")
        else:
            raise PreprocessNonDataException(f"Unsupported primitive type: {type(data)}")
        
        
@beartype
@recursive_apply
def clip_color(data: Union[DataAbstractBase, Any]) -> DataAbstractBase:
    if type(data) == PointCloud:
        if data.is_empty:
            return data
        else:
            assert data.colors.ndim == 2 and data.colors.shape[-1] == 3, f"{data.colors.shape}" # (N,3)
            return data.new(
                colors = torch.clip(data.colors, min=torch.tensor([0.], device=data.colors.device, dtype=data.colors.dtype), max=torch.tensor([1.], device=data.colors.device, dtype=data.colors.dtype))
            )
    else:
        if isinstance(data, DataAbstractBase):
            raise PreprocessDataTypeException(f"Unsupported data type: {type(data)}")
        else:
            raise PreprocessNonDataException(f"Unsupported primitive type: {type(data)}")
        
        
        


# def normalize_color(data, color_mean: torch.Tensor, color_std: torch.Tensor):
#     if data is None:
#         return None
#     elif type(data) == DemoSequence:
#         demo_seq = [normalize_color(demo, color_mean=color_mean, color_std=color_std) for demo in data]
#         return DemoSequence(demo_seq = demo_seq, device=data.device)
#     elif type(data) == TargetPoseDemo:
#         scene_pc = normalize_color(data.scene_pc, color_mean=color_mean, color_std=color_std)
#         grasp_pc = normalize_color(data.grasp_pc, color_mean=color_mean, color_std=color_std)
#         target_poses = normalize_color(data.target_poses, color_mean=color_mean, color_std=color_std)
#         return TargetPoseDemo(scene_pc=scene_pc, grasp_pc=grasp_pc, target_poses=target_poses, device=data.device, name=data.name)
#     elif type(data) == PointCloud:
#         if data.is_empty():
#             return data
#         else:
#             return normalize_pc_color(data=data, color_mean=color_mean, color_std=color_std)
#     elif type(data) == SE3:
#         return data
#     else:
#         raise TypeError(f"Unknown data type ({type(data)}) is given.")



# def apply_SE3(data, poses: SE3, apply_right: bool = False):
#     if data is None:
#         return None
#     elif type(data) == DemoSequence:
#         demo_seq = [apply_SE3(data=demo, poses=poses, apply_right=apply_right) for demo in data]
#         return DemoSequence(demo_seq = demo_seq, device=data.device)
#     elif type(data) == TargetPoseDemo:
#         scene_pc = apply_SE3(data=scene_pc, poses=poses, apply_right=apply_right)
#         grasp_pc = apply_SE3(data=grasp_pc, poses=poses, apply_right=apply_right)
#         target_poses = apply_SE3(data=target_poses, poses=poses, apply_right=apply_right)
#         return TargetPoseDemo(scene_pc=scene_pc, grasp_pc=grasp_pc, target_poses=target_poses, device=data.device, name=data.name)
#     elif type(data) == PointCloud:
#         if data.is_empty():
#             return data
#         else:
#             return data.transformed(poses)
#     elif type(data) == SE3:
#         return SE3.multiply(poses, data, apply_right=apply_right)
#     else:
#         raise TypeError(f"Unknown data type ({type(data)}) is given.")
    


# def jitter_points(data, jitter_std: float):
#     if data is None:
#         return None
#     elif type(data) == DemoSequence:
#         demo_seq = [jitter_points(data=demo, jitter_std=jitter_std) for demo in data]
#         return DemoSequence(demo_seq = demo_seq, device=data.device)
#     elif type(data) == TargetPoseDemo:
#         scene_pc = jitter_points(data=scene_pc, jitter_std=jitter_std)
#         grasp_pc = jitter_points(data=grasp_pc, jitter_std=jitter_std)
#         target_poses = jitter_points(data=target_poses, jitter_std=jitter_std)
#         return TargetPoseDemo(scene_pc=scene_pc, grasp_pc=grasp_pc, target_poses=target_poses, device=data.device, name=data.name)
#     elif type(data) == PointCloud:
#         if data.is_empty():
#             return data
#         else:
#             points = data.points + torch.randn(*(data.points.shape), device=data.points.device, dtype=data.points.dtype) * jitter_std
#             return PointCloud(points=points, colors=data.colors)
#     elif type(data) == SE3:
#         return data
#     else:
#         raise TypeError(f"Unknown data type ({type(data)}) is given.")



# def jitter_colors(data, jitter_std: float, cutoff: bool = False):
#     if data is None:
#         return None
#     elif type(data) == DemoSequence:
#         demo_seq = [jitter_points(data=demo, jitter_std=jitter_std, cutoff=cutoff) for demo in data]
#         return DemoSequence(demo_seq = demo_seq, device=data.device)
#     elif type(data) == TargetPoseDemo:
#         scene_pc = jitter_points(data=scene_pc, jitter_std=jitter_std, cutoff=cutoff)
#         grasp_pc = jitter_points(data=grasp_pc, jitter_std=jitter_std, cutoff=cutoff)
#         target_poses = jitter_points(data=target_poses, jitter_std=jitter_std, cutoff=cutoff)
#         return TargetPoseDemo(scene_pc=scene_pc, grasp_pc=grasp_pc, target_poses=target_poses, device=data.device, name=data.name)
#     elif type(data) == PointCloud:
#         if data.is_empty():
#             return data
#         else:
#             colors = data.colors + torch.randn(*(data.colors.shape), device=data.colors.device, dtype=data.colors.dtype) * jitter_std
#             if cutoff:
#                 colors = torch.where(colors > 1., torch.tensor(1., dtype=colors.dtype, device=colors.device), colors)
#                 colors = torch.where(colors < 0., torch.tensor(0., dtype=colors.dtype, device=colors.device), colors)
#             return PointCloud(points=data.points, colors=colors)
#     elif type(data) == SE3:
#         return data
#     else:
#         raise TypeError(f"Unknown data type ({type(data)}) is given.")



# class EdfTransform(torch.nn.Module):
#     def __init__(self, supported_type):
#         super().__init__()
#         self.supported_type = supported_type

#     def forward(self, data):
#         if type(data) not in self.supported_type:
#             raise TypeError
#         return data



# class Rescale(EdfTransform):
#     def __init__(self, rescale_factor: float) -> None:
#         super().__init__(supported_type = [None, PointCloud, SE3, TargetPoseDemo, DemoSequence])
#         self.rescale_factor: float = rescale_factor

#     def forward(self, data):
#         data = super().forward(data)
#         return rescale(data=data, rescale_factor = self.rescale_factor)
        

    
# class NormalizeColor(EdfTransform):
#     def __init__(self, color_mean: torch.Tensor, color_std: torch.Tensor) -> None:
#         super().__init__(supported_type = [None, PointCloud, SE3, TargetPoseDemo, DemoSequence])
#         self.color_mean: torch.Tensor = color_mean
#         self.color_std: torch.Tensor = color_std

#     def forward(self, data):
#         data = super().forward(data)
#         return normalize_color(data=data, color_mean = self.color_mean, color_std = self.color_std)


    
# class Downsample(EdfTransform):
#     def __init__(self, voxel_size: float, coord_reduction: str = "average") -> None:
#         super().__init__(supported_type = [None, PointCloud, SE3, TargetPoseDemo, DemoSequence])
#         self.voxel_size: float = voxel_size
#         self.coord_reduction: str = coord_reduction

#     def forward(self, data):
#         data = super().forward(data)
#         return downsample(data=data, voxel_size=self.voxel_size, coord_reduction=self.coord_reduction)
    
#     def extra_repr(self) -> str:
#         return f"voxel_size: {self.voxel_size}, coord_reduction: {self.coord_reduction}"


    
# class ApplySE3(EdfTransform):
#     def __init__(self, poses: SE3, apply_right: bool = False) -> None:
#         super().__init__(supported_type = [None, PointCloud, SE3, TargetPoseDemo, DemoSequence])
#         self.poses = SE3(poses=poses.detach().clone())
#         self.apply_right = apply_right

#     def forward(self, data):
#         data = super().forward(data)
#         return apply_SE3(data=data, poses=self.poses, apply_right=self.apply_right)



# class PointJitter(EdfTransform):
#     def __init__(self, jitter_std: float) -> None:
#         super().__init__(supported_type = [None, PointCloud, SE3, TargetPoseDemo, DemoSequence])
#         self.jitter_std: float = jitter_std

#     def forward(self, data):
#         data = super().forward(data)
#         return jitter_points(data=data, jitter_std=self.jitter_std)
    


# class ColorJitter(EdfTransform):
#     def __init__(self, jitter_std: float, cutoff: bool = False) -> None:
#         super().__init__(supported_type = [None, PointCloud, SE3, TargetPoseDemo, DemoSequence])
#         self.jitter_std: float = jitter_std
#         self.cutoff: bool = cutoff

#     def forward(self, data):
#         data = super().forward(data)
#         return jitter_colors(data=data, jitter_std=self.jitter_std, cutoff=self.cutoff)