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
from tools.make_axis_align_real_data import (
    align_colmap_pose_to_arkit_coord,
    tracking_quality_filter,
    decompose_to_sRT,
)
from utils.util import read_json, write_json

from pyrender import (
    PerspectiveCamera,
    Mesh,
    Node,
    Scene,
    Viewer,
    OffscreenRenderer,
    RenderFlags,
)

# def render_depth(obj)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arkit_raw_dir",
        # default="/home/ybbbbt/Developer/neural_scene/data/arkit_recon/arkit_box_2",
    )
    parser.add_argument(
        "--obj_in_arkit_coord",
        # default="/home/ybbbbt/Developer/neural_scene/data/object_capture_recon/box/obj_in_colmap_coord.obj",
    )
    parser.add_argument("--colmap_refine_dir")
    """
    Tune with MeshLab: Filters -> Mesh Layers -> Matrix: set from translation/rotaton/scale
    """
    parser.add_argument("--x_rot", default=-90, type=float)  # X rotation in meshlab
    parser.add_argument("--y_rot", default=0, type=float)  # Y rotation in meshlab
    parser.add_argument(
        "--output_dir", default="debug/processed_real_data_bg", type=str
    )  # Y rotation in meshlab

    args = parser.parse_args()

    visualizer = O3dVisualizer()

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.5, origin=[0, 0, 0]
    )
    visualizer.add_o3d_geometry(mesh_frame)

    # rotate mesh
    mesh = o3d.io.read_triangle_mesh(args.obj_in_arkit_coord)
    from scipy.spatial.transform import Rotation as R

    # make axis align
    rotation = R.from_euler("xyz", [-args.x_rot, 0, 0], degrees=True).as_matrix()
    rotation = (
        rotation @ R.from_euler("xyz", [0, -args.y_rot, 0], degrees=True).as_matrix()
    )
    mesh.rotate(rotation, center=(0, 0, 0))

    # transform mat for frames
    transform_mat = np.eye(4)
    transform_mat[:3, :3] = rotation
    # transform_mat[:3, 3] = translate

    visualizer.add_o3d_geometry(mesh)
    # visualizer.run_visualize()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    o3d.io.write_triangle_mesh(os.path.join(output_dir, "aligned.obj"), mesh)

    # initialize mask render
    obj_trimesh = trimesh.load(os.path.join(output_dir, "aligned.obj"))
    obj_mesh = Mesh.from_trimesh(obj_trimesh)
    scene = Scene(ambient_light=np.array([0.5, 0.5, 0.5, 1.0]))
    obj_node = Node(mesh=obj_mesh, translation=np.zeros(3))
    scene.add_node(obj_node)

    # pre frame processing
    frame_info = {"frames": []}
    arkit_raw_dir = args.arkit_raw_dir
    all_image_files = sorted(glob.glob(arkit_raw_dir + "/frame_*.jpg"))

    colmap_refined_frames = read_json(
        # os.path.join(args.colmap_refine_dir, "nerfpp_fmt", "nerfpp_cameras.json")
        os.path.join(args.colmap_refine_dir, "posed_images", "nerfpp_cameras.json")
    )

    # align colmap to arkit pose
    transform_colmap_to_arkit = np.eye(4)
    transform_colmap_to_arkit, colmap_refined_frames = align_colmap_pose_to_arkit_coord(
        colmap_refined_frames, all_image_files
    )
    s, R, t = decompose_to_sRT(transform_colmap_to_arkit)
    print(s, R, t)

    tracking_quality_th = 1.1
    tracking_quality_th = tracking_quality_filter(all_image_files, drop_ratio=20)
    print("tracking quality threshold", tracking_quality_th)

    transform_info = {
        "transform_colmap_to_arkit_sRT": transform_colmap_to_arkit.tolist(),
        "transform_alignment": transform_mat.tolist(),
    }
    write_json(transform_info, os.path.join(output_dir, "transform_info.json"))

    os.makedirs(os.path.join(output_dir, "full"), exist_ok=True)

    for idx in tqdm(range(len(all_image_files))):
        absolute_img_name = all_image_files[idx]
        img_name = os.path.basename(absolute_img_name)
        arkit_frame_info = read_json(
            os.path.join(arkit_raw_dir, img_name[:-3] + "json")
        )

        if idx == 0:
            h, w, _ = imageio.imread(absolute_img_name).shape
            # write camera angle
            intrinsics = np.array(arkit_frame_info["intrinsics"])
            focal, cx, cy = intrinsics[0], intrinsics[2], intrinsics[5]
            xfov = np.arctan(w / 2 / focal) * 2
            print("xfov =", xfov)
            frame_info["camera_angle_x"] = xfov

            render = OffscreenRenderer(viewport_width=w, viewport_height=h)
            yfov = np.arctan(h / 2 / focal) * 2
            cam = PerspectiveCamera(yfov=yfov)
            cam_node = scene.add(cam)

            camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
                1920, 1440, focal, focal, cx, cy
            )
            tsdf_cubic_size = 3.0
            voxel_length = tsdf_cubic_size / 512
            trunc_margin = 0.04
            volume = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=voxel_length,
                sdf_trunc=trunc_margin,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
            )

        if arkit_frame_info["motionQuality"] < tracking_quality_th:
            continue

        if img_name not in colmap_refined_frames:
            continue

        # read pose from arkit
        # pose_ndc = np.array(arkit_frame_info["cameraPoseARFrame"]).reshape(4, 4)

        # read pose from colmap , and convert to ndc coordinate
        pose_ndc = np.array(colmap_refined_frames[img_name]["W2C"]).reshape(4, 4)
        pose_ndc = np.linalg.inv(pose_ndc)
        fix_rot = np.array([1, 0, 0, 0, -1, 0, 0, 0, -1]).reshape(3, 3)
        pose_ndc[:3, :3] = pose_ndc[:3, :3] @ fix_rot

        # transform to arkit pose space
        s, R, t = decompose_to_sRT(transform_colmap_to_arkit)
        # pose_ndc = transform_colmap_to_arkit @ pose_ndc
        # print(s, R, t)
        pose_ndc[:3, 3] = R @ (pose_ndc[:3, 3] * s) + t
        pose_ndc[:3, :3] = R @ pose_ndc[:3, :3]

        # apply alignment to poses
        pose_ndc = transform_mat @ pose_ndc
        color_o3d = o3d.io.read_image(absolute_img_name)
        sensor_depth = cv2.imread(
            os.path.join(arkit_raw_dir, f"depth_{img_name[6:11]}.png"), -1
        )
        sensor_depth = cv2.resize(
            sensor_depth, dsize=(w, h), interpolation=cv2.INTER_NEAREST
        )
        sensor_depth = sensor_depth.astype(np.float32) * 1e-3
        cv2.imwrite("/tmp/depth.png", (sensor_depth * 1000).astype(np.uint16))
        # need same size as rgb
        depth_o3d = o3d.io.read_image("/tmp/depth.png")
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d,
            depth_o3d,
            depth_trunc=4.0,
            convert_rgb_to_intensity=False,
        )
        pose_twc = pose_ndc
        pose_twc[:3, :3] = pose_ndc[:3, :3] @ fix_rot
        volume.integrate(
            rgbd,
            camera_intrinsics,
            np.linalg.inv(pose_twc),
        )

        curr_frame_info = {
            "idx": idx,
            "transform_matrix": pose_ndc.tolist(),
            "file_path": f"./full/{img_name[:-4]}",
        }
        frame_info["frames"].append(curr_frame_info)

    write_json(frame_info, os.path.join(output_dir, "transforms_full.json"))
    mesh = volume.extract_triangle_mesh()
    o3d.io.write_triangle_mesh("debug/mesh.ply", mesh)
