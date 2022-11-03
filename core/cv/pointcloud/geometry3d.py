import open3d as o3d
import numpy as np
from PIL import Image


def array2pcd(points, colors):
    """
    Convert points and colors into open3d point cloud.

    Args:
        points(np.array): coordinates of the points.
        colors(np.array): RGB values of the points.

    Returns:
        open3d.geometry.PointCloud: the point cloud.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def pcd2array(pcd):
    """
    Convert open3d point cloud into points and colors.

    Args:
        pcd(open3d.geometry.PointCloud): the point cloud.


    Returns:
        np.array, np.array: coordinates of the points, RGB values of the points.
    """
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    return points, colors


def merge_pcds(pcds):
    """
    Merge several point cloud into a single one.

    Args:
        pcds(list): list of point cloud.

    Returns:
        open3d.geometry.PointCloud: the merged point cloud.
    """
    ret_pcd = o3d.geometry.PointCloud()
    if len(pcds) == 0:
        return ret_pcd
    old_points, old_colors = pcd2array(pcds[0])
    for i in range(1, len(pcds)):
        points, colors = pcd2array(pcds[i])
        old_points = np.concatenate((old_points, points))
        old_colors = np.concatenate((old_colors, colors))
    return array2pcd(old_points, old_colors)


def generate_scene_pointcloud(depth, rgb, intrinsics, depth_scale, use_mask=True):
    '''Generate point cloud from depth image and color image

    Args:
        depth(str / np.array): Depth image path or depth.
        rgb(str / np.array): RGB image path or RGB values.
        intrinsics(np.array): Camera intrinsics matrix.
        depth_scale(float): The depth factor.
        use_mask(bool): Whether to use mask for pointcloud.

    Returns:
        open3d.geometry.PointCloud: the point cloud
    '''
    if type(depth) == str and type(rgb) == str:
        colors = np.array(Image.open(rgb), dtype=np.float32) / 255.0
        depths = np.array(Image.open(depth))

    elif type(depth) == np.ndarray and type(rgb) == np.ndarray:
        colors = rgb
        depths = depth

    else:
        raise ValueError('The type of depth and rgb must be str or np.ndarray')

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    xmap, ymap = np.arange(colors.shape[1]), np.arange(colors.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)

    points_z = depths / depth_scale
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z

    mask = (points_z > 0)
    points = np.stack([points_x, points_y, points_z], axis=-1)
    if use_mask:
        points = points[mask]
        colors = colors[mask]
    else:
        points = points.reshape((-1, 3))
        colors = colors.reshape((-1, 3))
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)

    return cloud