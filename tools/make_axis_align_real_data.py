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
from scipy.spatial.transform import Rotation as R

from pyrender import (
    PerspectiveCamera,
    Mesh,
    Node,
    Scene,
    Viewer,
    OffscreenRenderer,
    RenderFlags,
)


def decompose_to_sRT(Trans):
    t = Trans[:3, 3]
    R = Trans[:3, :3]
    # assume x y z have the same scale
    scale = np.linalg.norm(R[:3, 0])
    R = R / scale
    return scale, R, t


def align_colmap_pose_to_arkit_coord(
    colmap_refined_frames,
    arkit_all_image_files,
    use_ransac_filter=True,
    # use_ransac_filter=False,
):
    colmap_centers = []
    arkit_centers = []
    overlap_image_names = []
    for absolute_image_name in arkit_all_image_files:
        img_name = os.path.basename(absolute_image_name)
        if img_name not in colmap_refined_frames:
            continue
        overlap_image_names += [img_name]
        arkit_frame_info = read_json(absolute_image_name[:-3] + "json")
        pose_ndc = np.array(arkit_frame_info["cameraPoseARFrame"]).reshape(4, 4)
        arkit_centers += [pose_ndc[:3, 3]]
        pose_colmap = np.array(colmap_refined_frames[img_name]["W2C"]).reshape(
            4, 4
        )  # Tcw
        pose_colmap = np.linalg.inv(pose_colmap)
        colmap_centers.append(pose_colmap[:3, 3])

    colmap_centers = np.stack(colmap_centers, axis=0)
    arkit_centers = np.stack(arkit_centers, axis=0)

    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(colmap_centers)

    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(arkit_centers)

    if use_ransac_filter:
        corr = np.arange(colmap_centers.shape[0])
        corr = np.stack([corr, corr], axis=1)
        # using ransac to filter bad poses
        result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
            source,
            target,
            o3d.utility.Vector2iVector(corr),
            0.2,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(True),
        )
        transformation = result.transformation

        # filter by resulting correspondence
        remaining_corr = np.asarray(result.correspondence_set)
        for i, name in enumerate(overlap_image_names):
            if i not in remaining_corr:
                print("Remove bad frame", name)
                del colmap_refined_frames[name]
    else:
        p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        p2p.with_scaling = True
        corr = np.arange(colmap_centers.shape[0])
        corr = np.stack([corr, corr], axis=1)
        transformation = p2p.compute_transformation(
            source, target, o3d.utility.Vector2iVector(corr)
        )

    return transformation, colmap_refined_frames


def read_sense_frame_txt(pose_path):
    pose_dict = {}
    with open(pose_path) as file:
        lines = file.readlines()
        lines = lines[4:]
        for line in lines:
            fname, tx, ty, tz, qx, qy, qz, qw = line.strip().split(" ")
            fname += ".jpg"
            pose = np.eye(4)
            pose[0, 3] = tx
            pose[1, 3] = ty
            pose[2, 3] = tz
            pose[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
            pose = np.linalg.inv(pose)
            pose_dict[fname] = {"W2C": pose}
            # print(fname, pose)
    return pose_dict


def tracking_quality_filter(arkit_all_image_files, drop_ratio=50.0):
    """
    drop frames with bad quality
    """
    qualities = []
    for absolute_image_name in arkit_all_image_files:
        arkit_frame_info = read_json(absolute_image_name[:-3] + "json")
        qualities += [arkit_frame_info["motionQuality"]]
    qualities = np.array(qualities)
    quality_th = np.percentile(qualities, drop_ratio)
    return quality_th


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arkit_raw_dir",
        default="/home/ybbbbt/Developer/neural_scene/data/arkit_recon/arkit_box_2",
    )
    parser.add_argument(
        "--obj_in_colmap_coord",
        default="/home/ybbbbt/Developer/neural_scene/data/object_capture_recon/box/obj_in_colmap_coord.obj",
    )
    parser.add_argument("--colmap_refine_dir")
    """
    Tune with MeshLab: Filters -> Mesh Layers -> Matrix: set from translation/rotaton/scale
    """
    parser.add_argument("--x_rot", default=-90, type=float)  # X rotation in meshlab
    parser.add_argument("--y_rot", default=0, type=float)  # Y rotation in meshlab
    parser.add_argument(
        "--output_dir", default="debug/processed_real_data", type=str
    )  # Y rotation in meshlab
    parser.add_argument(
        "--instance_id_for_mask", default=1, type=int
    )  # X rotation in meshlab

    args = parser.parse_args()

    mode = "object_capture_aligned_to_colmap"
    # mode = "sense"

    visualizer = O3dVisualizer()

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.5, origin=[0, 0, 0]
    )
    visualizer.add_o3d_geometry(mesh_frame)

    # read frame info
    arkit_raw_dir = args.arkit_raw_dir
    all_image_files = sorted(glob.glob(arkit_raw_dir + "/frame_*.jpg"))
    if mode == "sense":
        colmap_refined_frames = read_sense_frame_txt(
            os.path.join(args.colmap_refine_dir, "pose.txt")
        )
    else:
        colmap_refined_frames = read_json(
            # os.path.join(args.colmap_refine_dir, "nerfpp_fmt", "nerfpp_cameras.json")
            # os.path.join(args.colmap_refine_dir, "posed_images", "nerfpp_cameras.json")
            os.path.join(
                args.colmap_refine_dir, "output/posed_images", "nerfpp_cameras.json"
            )
        )

    # align colmap to arkit pose
    transform_colmap_to_arkit = np.eye(4)
    transform_colmap_to_arkit, colmap_refined_frames = align_colmap_pose_to_arkit_coord(
        colmap_refined_frames, all_image_files
    )
    s, R, t = decompose_to_sRT(transform_colmap_to_arkit)
    print(s, R, t)

    # read and process mesh
    mesh = o3d.io.read_triangle_mesh(args.obj_in_colmap_coord)

    # if mode == "sense":
    mesk = mesh.transform(transform_colmap_to_arkit)

    # rotate mesh
    from scipy.spatial.transform import Rotation as R

    # make axis align
    rotation = R.from_euler("xyz", [-args.x_rot, 0, 0], degrees=True).as_matrix()
    rotation = (
        rotation @ R.from_euler("xyz", [0, -args.y_rot, 0], degrees=True).as_matrix()
    )
    mesh.rotate(rotation, center=(0, 0, 0))

    # translate to make bbox center at origin
    translate = -mesh.get_axis_aligned_bounding_box().get_center()
    mesh.translate(translate)

    # compute mesh bbox
    bbox = mesh.get_axis_aligned_bounding_box()
    bound = np.array([bbox.min_bound, bbox.max_bound])
    size = bound[1] - bound[0]

    # transform mat for frames
    transform_mat = np.eye(4)
    transform_mat[:3, :3] = rotation
    transform_mat[:3, 3] = translate

    visualizer.add_o3d_geometry(mesh)
    # visualizer.run_visualize()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # write mesh and bbox info
    o3d.io.write_triangle_mesh(os.path.join(output_dir, "aligned.obj"), mesh)
    write_json(
        {
            "max_bound": bbox.max_bound.tolist(),
            "min_bound": bbox.min_bound.tolist(),
            "size": size.tolist(),
        },
        os.path.join(output_dir, "bbox.json"),
    )

    # initialize mask render
    obj_trimesh = trimesh.load(os.path.join(output_dir, "aligned.obj"))
    obj_mesh = Mesh.from_trimesh(obj_trimesh)
    scene = Scene(ambient_light=np.array([0.5, 0.5, 0.5, 1.0]))
    obj_node = Node(mesh=obj_mesh, translation=np.zeros(3))
    scene.add_node(obj_node)

    # pre frame processing
    frame_info = {"frames": []}

    tracking_quality_th = 1.1
    if args.instance_id_for_mask == 34:  # desk use larger drop ratio
        tracking_quality_th = tracking_quality_filter(all_image_files, drop_ratio=50)
    else:
        tracking_quality_th = tracking_quality_filter(all_image_files, drop_ratio=20)
    print("tracking quality threshold", tracking_quality_th)

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

        if arkit_frame_info["motionQuality"] < tracking_quality_th:
            continue
        if img_name not in colmap_refined_frames:
            continue
        # pose_ndc = np.array(arkit_frame_info["cameraPoseARFrame"]).reshape(4, 4)
        # read pose from colmap refined, and convert to ndc coordinate
        pose_ndc = np.array(colmap_refined_frames[img_name]["W2C"]).reshape(4, 4)
        pose_ndc = np.linalg.inv(pose_ndc)
        fix_rot = np.array([1, 0, 0, 0, -1, 0, 0, 0, -1]).reshape(3, 3)
        pose_ndc[:3, :3] = pose_ndc[:3, :3] @ fix_rot
        # transform to arkit pose
        s, R, t = decompose_to_sRT(transform_colmap_to_arkit)
        # pose_ndc = transform_colmap_to_arkit @ pose_ndc
        # print(s, R, t)
        pose_ndc[:3, 3] = R @ (pose_ndc[:3, 3] * s) + t
        pose_ndc[:3, :3] = R @ pose_ndc[:3, :3]
        # apply alignment to poses
        pose_ndc = transform_mat @ pose_ndc

        # render depth
        scene.set_pose(cam_node, pose_ndc)
        mesh_proj_color, rendered_depth = render.render(scene)

        # use sensor depth
        # sensor_depth = cv2.imread(
        #     os.path.join(arkit_raw_dir, f"depth_{img_name[6:11]}.png"), -1
        # )
        # sensor_depth = cv2.resize(
        #     sensor_depth, dsize=(w, h), interpolation=cv2.INTER_NEAREST
        # )
        # sensor_depth = sensor_depth.astype(np.float32) * 1e-3

        # cv2.imwrite(
        #     os.path.join(output_dir, "full", f"{img_name[:-4]}.depth.png"),
        #     (sensor_depth * 1000).astype(np.uint16),
        # )

        cv2.imwrite(
            os.path.join(output_dir, "full", f"{img_name[:-4]}.depth.png"),
            (rendered_depth * 1000).astype(np.uint16),
        )

        # compute normal
        normal_map = compute_normal_from_depth(
            rendered_depth.astype(np.float64), focal=focal
        )
        normal_map = (normal_map + 1) / 2 * 255
        normal_map[rendered_depth == 0] = 0
        imageio.imwrite(
            os.path.join(output_dir, "full", f"{img_name[:-4]}.normal.png"),
            normal_map.astype(np.uint8),
        )

        seg = np.zeros_like(rendered_depth)
        seg[rendered_depth != 0] = args.instance_id_for_mask
        cv2.imwrite(
            os.path.join(output_dir, "full", f"{img_name[:-4]}.seg.png"),
            seg.astype(np.uint16),
        )

        # include rgb file (to png)
        rgb = cv2.imwrite(
            os.path.join(output_dir, "full", f"{img_name[:-4]}.png"),
            cv2.imread(absolute_img_name),
        )

        curr_frame_info = {
            "idx": idx,
            "transform_matrix": pose_ndc.tolist(),
            "file_path": f"./full/{img_name[:-4]}",
        }
        frame_info["frames"].append(curr_frame_info)

    write_json(frame_info, os.path.join(output_dir, "transforms_full.json"))
