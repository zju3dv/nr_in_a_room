import numpy as np
import argparse
import sys
import os

sys.path.append(os.getcwd())  # noqa
import glob
import copy
import shutil
import json
from pathlib import Path
from PIL import Image
import open3d as o3d
from tools.O3dVisualizer import O3dVisualizer
import matplotlib.pyplot as plt

# from datasets.geo_utils import observe_angle_distance
from utils.util import *


def decompose_to_sRT(Trans):
    t = Trans[:3, 3]
    R = Trans[:3, :3]
    # assume x y z have the same scale
    scale = np.linalg.norm(R[:3, 0])
    R = R / scale
    return scale, R, t


def draw_poses(visualizer, pose_info_json):
    frame_num = len(pose_info_json["frames"])
    camera_centers = []
    lines_pt, lines_idx, lines_color = [], [], []

    idx = 0
    for frame_id, frame in enumerate(pose_info_json["frames"]):
        Twc = np.array(frame["transform_matrix"])
        s, R, t = decompose_to_sRT(Twc)
        print(s, R, t)
        # for nerf_synthetic, we need some transformation
        fix_rot = np.array([1, 0, 0, 0, -1, 0, 0, 0, -1]).reshape(3, 3)
        # fix_rot_blender_jy = np.array([0, 0, 1, -1, 0, 0, 0, -1, 0]).reshape(3, 3)
        # fix_rot_blender_jy = fix_rot_blender_jy @ fix_rot
        # print(fix_rot_blender_jy)
        fix_rot_blender_jy = np.array([0, 0, -1, -1, 0, 0, 0, 1, 0]).reshape(3, 3)

        # Twc[:3, :3] = Twc[:3, :3] @ fix_rot_blender_jy
        # -G, -B, R
        Twc[:3, :3] = Twc[:3, :3] @ fix_rot
        # Twc[:3, :3] = Twc[:3, :3]

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

    print("Camera center num:", len(camera_centers))
    # draw line via cylinder, which we can control the line thickness
    # visualizer.add_line_set(lines_pt, lines_idx, colors=lines_color, radius=0.003)

    # draw line via LineSet
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.array(lines_pt)),
        lines=o3d.utility.Vector2iVector(np.array(lines_idx)),
    )
    line_set.colors = o3d.utility.Vector3dVector(lines_color)
    visualizer.add_o3d_geometry(line_set)

    camera_centers = np.array(camera_centers)
    visualizer.add_np_points(
        camera_centers,
        color=map_to_color(np.arange(0, frame_num), cmap="plasma"),
        size=0.01,
    )


def draw_arkit_poses(visualizer: O3dVisualizer, recon_path):
    camera_centers = []
    lines_pt, lines_idx, lines_color = [], [], []

    idx = 0
    all_frame_json = sorted(glob.glob(recon_path + "/frame_*.json"))
    frame_num = len(all_frame_json)
    for frame_id, frame_json in enumerate(all_frame_json):
        frame_info = read_json(os.path.join(recon_path, frame_json))
        Twc = np.array(frame_info["cameraPoseARFrame"]).reshape(4, 4)
        # print(Twc)
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

    print("Camera center num:", len(camera_centers))
    # draw line via cylinder, which we can control the line thickness
    # visualizer.add_line_set(lines_pt, lines_idx, colors=lines_color, radius=0.003)

    # draw line via LineSet
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.array(lines_pt)),
        lines=o3d.utility.Vector2iVector(np.array(lines_idx)),
    )
    line_set.colors = o3d.utility.Vector3dVector(lines_color)
    visualizer.add_o3d_geometry(line_set)

    camera_centers = np.array(camera_centers)
    visualizer.add_np_points(
        camera_centers,
        color=map_to_color(np.arange(0, frame_num), cmap="plasma"),
        size=0.01,
    )


if __name__ == "__main__":
    # params
    parser = argparse.ArgumentParser()
    # data paths
    parser.add_argument("--pcd", default=None)
    parser.add_argument("--mesh", default=None)
    parser.add_argument("--transform_json", default=None)
    parser.add_argument("--arkit_recon", default=None)
    args = parser.parse_args()

    visualizer = O3dVisualizer()
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.1, origin=[0, 0, 0]
    )
    visualizer.add_o3d_geometry(mesh_frame)
    if args.pcd:
        pcd = o3d.io.read_point_cloud(args.pcd)
        visualizer.add_o3d_geometry(pcd)
    if args.mesh:
        mesh = o3d.io.read_triangle_mesh(args.mesh)
        mesh.compute_vertex_normals()
        if str(args.mesh)[-4:].lower() == ".glb":
            fix_rot = np.eye(4)
            fix_rot[:3, :3] = np.array([1, 0, 0, 0, 0, 1, 0, -1, 0]).reshape(3, 3)
            # fix_rot = np.linalg.inv(fix_rot)
            mesh = mesh.transform(fix_rot)
        visualizer.add_o3d_geometry(mesh)
    if args.transform_json:
        draw_poses(visualizer, read_json(args.transform_json))

    if args.arkit_recon:
        draw_arkit_poses(visualizer, args.arkit_recon)

    visualizer.run_visualize()
