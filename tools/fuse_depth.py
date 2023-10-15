import cv2
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
import pybullet as p
import matplotlib.pyplot as plt

# from datasets.geo_utils import observe_angle_distance
from utils.util import map_to_color, read_json
from scipy.spatial.transform import Rotation as R


def visualize_fused_depth(visualizer, pose_info_json, base_dir):
    frame_num = len(pose_info_json["frames"])
    camera_centers = []
    lines_pt, lines_idx, lines_color = [], [], []

    idx = 0
    for frame_id, frame in enumerate(pose_info_json["frames"]):
        Twc = np.array(frame["transform_matrix"])
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

        # depth
        depth_file = os.path.join(base_dir, f'{frame["file_path"]}.depth.png')
        depth = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH) * 1.0e-3
        H, W = depth.shape
        focal = (W / 2) / np.tan(pose_info_json["camera_angle_x"] / 2)
        src_K = [focal, focal, W / 2, H / 2]
        ys, xs = np.where(depth > 0)
        warped_depth = np.zeros_like(depth)
        pts = np.empty((len(xs), 3))
        pts[:, 0] = (xs - src_K[2]) / src_K[0]
        pts[:, 1] = (ys - src_K[3]) / src_K[1]
        pts[:, 2] = 1.0
        pts = pts * np.repeat(depth[ys, xs].reshape(-1, 1), 3, axis=1)
        # lift to space
        pts_w = np.transpose(np.matmul(Twc[:3, :3], np.transpose(pts))) + Twc[:3, 3]
        pts_w = pts_w[np.random.choice(pts_w.shape[0], 1000)]
        visualizer.add_np_points(pts_w, size=0.01)

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
    parser.add_argument("--transform_json", default=None)
    parser.add_argument("--base_dir", default=None)
    args = parser.parse_args()

    visualizer = O3dVisualizer()
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1.0, origin=[0, 0, 0]
    )
    if args.transform_json:
        visualize_fused_depth(visualizer, read_json(args.transform_json), args.base_dir)

    visualizer.run_visualize()
