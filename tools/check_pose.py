import numpy as np
import argparse
import sys
import os

sys.path.append(os.getcwd())  # noqa
from data_gen.data_geo_utils import create_sphere_lookat_poses
import open3d as o3d
from tools.O3dVisualizer import O3dVisualizer
import matplotlib.pyplot as plt

# from datasets.geo_utils import observe_angle_distance
from utils.util import *

# from render_tools.render_utils import *


def spheric_pose(theta, phi, radius, height):
    trans_t = lambda t: np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, -0.9 * t],
            [0, 0, 1, t],
            [0, 0, 0, 1],
        ]
    )

    rot_phi = lambda phi: np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(phi), -np.sin(phi), 0],
            [0, np.sin(phi), np.cos(phi), 0],
            [0, 0, 0, 1],
        ]
    )

    rot_theta = lambda th: np.array(
        [
            [np.cos(th), 0, -np.sin(th), 0],
            [0, 1, 0, 0],
            [np.sin(th), 0, np.cos(th), 0],
            [0, 0, 0, 1],
        ]
    )

    c2w = rot_theta(theta) @ rot_phi(phi) @ trans_t(radius)
    c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
    c2w[2, 3] += height
    return c2w[:3]


def create_spheric_poses(radius, downward_deg, height, n_poses):
    """
    Create circular poses around z axis.
    Inputs:
        radius: the (negative) height and the radius of the circle.

    Outputs:
        spheric_poses: (n_poses, 3, 4) the poses in the circular path
    """
    spheric_poses = []
    for th in np.linspace(0, 2 * np.pi, n_poses + 1)[:-1]:
        pose = np.eye(4)
        pose[:3, :4] = spheric_pose(th, -(downward_deg * np.pi / 180), radius, height)
        fix_rot = np.eye(4)
        fix_rot[:3, :3] = np.array([1, 0, 0, 0, 0, 1, 0, -1, 0]).reshape(3, 3)
        pose = fix_rot @ pose
        spheric_poses += [pose]  # 36 degree view downwards
    return np.stack(spheric_poses, 0)


def draw_poses(visualizer, poses):
    camera_centers = []
    lines_pt, lines_idx, lines_color = [], [], []

    idx = 0
    for frame_id, pose in enumerate(poses):
        Twc = pose
        # for nerf_synthetic, we need some transformation
        fix_rot = np.array([1, 0, 0, 0, -1, 0, 0, 0, -1]).reshape(3, 3)
        Twc[:3, :3] = Twc[:3, :3] @ fix_rot

        center = Twc[:3, 3]
        camera_centers.append(center)
        # draw axis
        # RGB -> right, down, forward
        axis_size = 0.1
        # for .T, you can follow https://stackoverflow.com/questions/12148351/
        axis_pts = (Twc[:3, :3] @ (np.eye(3) * axis_size)).T + center
        lines_pt += [center, axis_pts[0, :], axis_pts[1, :], axis_pts[2, :]]
        lines_idx += [
            [idx * 4 + 0, idx * 4 + 1],
            [idx * 4 + 0, idx * 4 + 2],
            [idx * 4 + 0, idx * 4 + 3],
        ]
        lines_color += [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        idx += 1

    # draw line via cylinder, which we can control the line thickness
    visualizer.add_line_set(lines_pt, lines_idx, colors=lines_color, radius=0.003)

    # draw line via LineSet
    # line_set = o3d.geometry.LineSet(
    #     points=o3d.utility.Vector3dVector(np.array(lines_pt)),
    #     lines=o3d.utility.Vector2iVector(np.array(lines_idx)),
    # )
    # line_set.colors = o3d.utility.Vector3dVector(lines_color)
    # visualizer.add_o3d_geometry(line_set)

    camera_centers = np.array(camera_centers)
    visualizer.add_np_points(
        camera_centers,
        color=map_to_color(np.arange(0, len(poses)), cmap="plasma"),
        size=0.01,
    )


if __name__ == "__main__":
    # params
    parser = argparse.ArgumentParser()
    # data paths
    parser.add_argument("--pcd", default=None)
    args = parser.parse_args()

    visualizer = O3dVisualizer()
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1.0, origin=[0, 0, 0]
    )
    visualizer.add_o3d_geometry(mesh_frame)
    if args.pcd:
        pcd = o3d.io.read_point_cloud(args.pcd)
        visualizer.add_o3d_geometry(pcd)

    # poses = [
    #     np.array([0.9848, -0.1205,  0.1251,  0.1315,
    #              0.1733,  0.6342, -0.7534, -1.3516,
    #              0.0114,  0.7637,  0.6455,  0.9235]).reshape(3, 4),
    #     np.array([0.9848, -0.1205,  0.1251,  0.1315,
    #              0.1733,  0.6342, -0.7534, -1.3516,
    #              0.0114,  0.7637,  0.6455,  0.9235]).reshape(3, 4)
    # ]
    # draw_poses(visualizer, poses)

    # poses = create_spheric_poses(
    #     radius=1.0, downward_deg=45, height=0.8, n_poses=10)
    poses, _ = create_sphere_lookat_poses(2.0, 100, 3)

    # import ipdb

    # ipdb.set_trace()

    # poses = [poses[40], poses[41]]

    draw_poses(visualizer, poses)

    visualizer.run_visualize()
