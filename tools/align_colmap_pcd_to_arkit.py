import os
import sys
import pyglet

sys.path.append(".")  # noqa
pyglet.options["shadow_window"] = False
import argparse
import numpy as np
import torch
import json
import imageio
import cv2
import shutil
import glob
import open3d as o3d
from tqdm import tqdm
import trimesh
import matplotlib.pyplot as plt

from tools.O3dVisualizer import O3dVisualizer
from tools.apply_light_map_2d import compute_normal_from_depth
from utils.util import read_json, write_json
from tools.make_axis_align_real_data import (
    align_colmap_pose_to_arkit_coord,
    decompose_to_sRT,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arkit_raw_dir",
        default="/home/ybbbbt/Developer/neural_scene/data/arkit_recon/arkit_box_2",
    )
    parser.add_argument("--colmap_refine_dir")

    args = parser.parse_args()

    visualizer = O3dVisualizer()

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.5, origin=[0, 0, 0]
    )
    visualizer.add_o3d_geometry(mesh_frame)

    # pre frame processing
    frame_info = {"frames": []}
    arkit_raw_dir = args.arkit_raw_dir
    all_image_files = sorted(glob.glob(arkit_raw_dir + "/frame_*.jpg"))

    colmap_refined_frames = read_json(
        os.path.join(args.colmap_refine_dir, "posed_images", "nerfpp_cameras.json")
    )

    # align colmap to arkit pose
    transform_colmap_to_arkit = np.eye(4)
    transform_colmap_to_arkit, colmap_refined_frames = align_colmap_pose_to_arkit_coord(
        colmap_refined_frames, all_image_files
    )
    s, R, t = decompose_to_sRT(transform_colmap_to_arkit)
    print(s, R, t)

    colmap_pcd = o3d.io.read_point_cloud(
        os.path.join(args.colmap_refine_dir, "posed_images", "nerfpp_points.ply")
    )
    colmap_pcd = colmap_pcd.transform(transform_colmap_to_arkit)
    o3d.io.write_point_cloud(
        os.path.join(
            args.colmap_refine_dir, "posed_images", "colmap_point_in_arkit_coord.ply"
        ),
        colmap_pcd,
    )
