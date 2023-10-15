import ipdb
import numpy as np
import argparse
import sys
import os
import random

from numpy.lib.financial import _ipmt_dispatcher
from tqdm.std import tqdm

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
from utils.util import map_to_color, read_json, write_json
from scipy.spatial.transform import Rotation as R

SEED = 123
# torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


def draw_poses(visualizer, poses):
    camera_centers = []
    lines_pt, lines_idx, lines_color = [], [], []

    idx = 0
    for frame_id, pose in enumerate(poses):
        Twc = pose
        # for nerf_synthetic, we need some transformation
        fix_rot = np.array([1, 0, 0, 0, -1, 0, 0, 0, -1]).reshape(3, 3)
        # Twc[:3, :3] = Twc[:3, :3] @ fix_rot

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

    camera_centers = np.array(camera_centers)
    if len(poses) > 1:
        visualizer.add_np_points(
            camera_centers,
            color=map_to_color(np.arange(0, len(poses)), cmap="plasma"),
            size=0.01,
        )


def draw_poses_from_transform_json(visualizer, pose_info_json):
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


not_attach_floor_type = [
    "window",
    "picture",
    "monitor",
    "table_lamp",
]


def draw_ig_scene(visualizer, ig_scene_json, ig_scene_path):
    import pybullet as p

    # connect pybullet
    cid = p.connect(p.DIRECT)

    ig_scene_info = read_json(ig_scene_json)
    scene_name = ig_scene_info["scene"]
    # add scene wall visualization
    wall_obj_path = os.path.join(
        ig_scene_path, "scenes", scene_name, "shape/visual", "wall_vm.obj"
    )
    wall_mesh = o3d.io.read_triangle_mesh(wall_obj_path, True)
    visualizer.add_o3d_geometry(wall_mesh)

    lines_pt, lines_idx, lines_color = [], [], []
    text_pts = []
    i = 0
    for idx_b, obj in enumerate(ig_scene_info["objs"]):
        bbox3d = obj["bdb3d"]
        id_str = str(obj["id"])
        obj_type = obj["classname"]
        print(id_str)

        length = np.array(bbox3d["size"]) * 0.5
        # bbox center
        center = np.array(bbox3d["centroid"])
        # position of object
        scale = np.array(bbox3d["scale"])
        rotation = np.array(bbox3d["basis"]).reshape(3, 3)

        obj_meta_info = read_json(
            os.path.join(
                ig_scene_path, "objects", obj["model_path"], "misc/metadata.json"
            )
        )
        base_link_offset = np.array(obj_meta_info["base_link_offset"])
        obj_path = os.path.join(
            ig_scene_path, "objects", obj["model_path"], "shape/visual"
        )
        obj_id = obj["model_path"].split("/")[-1]
        urdf_file = os.path.join(
            ig_scene_path, "objects", obj["model_path"], "{}.urdf".format(obj_id)
        )

        # parse urdf file to obtain origin and filename
        print(urdf_file)
        # we follow load_articulated_object_in_renderer() in iGibson to correct the object pose
        object_pb_id = p.loadURDF(urdf_file, flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL)
        for shape in p.getVisualShapeData(object_pb_id):
            id, link_id, type, dimensions, filename, rel_pos, rel_orn, color = shape[:8]
            if link_id == -1:
                pos, orn = p.getBasePositionAndOrientation(object_pb_id)
            else:
                _, _, _, _, pos, orn = p.getLinkState(object_pb_id, link_id)

            rel_orn = R.from_quat(rel_orn).as_matrix()
            orn = R.from_quat(orn).as_matrix()

            mesh = o3d.io.read_triangle_mesh(filename, True)
            # remove base link offset
            vertices = np.asarray(mesh.vertices) - base_link_offset
            # apply initial pose offset
            vertices = vertices.dot(rel_orn[:3, :3].T) + rel_pos
            vertices = vertices.dot(orn[:3, :3].T) + pos
            # scale object and transform to the correct place
            vertices = (rotation @ (vertices * scale).T).T + center
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            mesh.compute_vertex_normals()
            visualizer.add_o3d_geometry(mesh)

        # draw semantic id
        text_pos = np.array(center[0:2].tolist() + [0]) + np.array([0, 0, 3.0])
        curr_text_pts = visualizer.text_3d(
            id_str, text_pos, font="/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf"
        )
        text_pts.append(np.asarray(curr_text_pts.points))

        curr_lines_pt = [
            np.array([-1, -1, -1]),
            np.array([1, -1, -1]),
            np.array([-1, 1, -1]),
            np.array([1, 1, -1]),
            np.array([-1, -1, 1]),
            np.array([1, -1, 1]),
            np.array([-1, 1, 1]),
            np.array([1, 1, 1]),
        ]
        curr_lines_pt = [center + rotation @ (length * x) for x in curr_lines_pt]

        prob_pt = np.array([0, 1, 0])
        prob_pt = center + rotation @ (length * prob_pt)
        visualizer.add_np_points(prob_pt.reshape(-1, 3), color=[1, 0, 0], size=0.05)

        lines_pt += curr_lines_pt
        lines_idx += [
            [i * 8 + 0, i * 8 + 1],
            [i * 8 + 0, i * 8 + 2],
            [i * 8 + 1, i * 8 + 3],
            [i * 8 + 2, i * 8 + 3],
            [i * 8 + 4, i * 8 + 5],
            [i * 8 + 4, i * 8 + 6],
            [i * 8 + 5, i * 8 + 7],
            [i * 8 + 6, i * 8 + 7],
            [i * 8 + 0, i * 8 + 4],
            [i * 8 + 1, i * 8 + 5],
            [i * 8 + 2, i * 8 + 6],
            [i * 8 + 3, i * 8 + 7],
        ]
        lines_color.extend([np.array([0, 1, 0]) for x in range(12)])
        i += 1

    lines_pt = np.array(lines_pt)

    visualizer.add_line_set(lines_pt, lines_idx, colors=lines_color, radius=0.01)

    text_pts = np.concatenate(text_pts, axis=0)
    visualizer.add_np_points(text_pts, color=[1, 0, 0], size=0.01)


ref_camera_center_list = []


def read_camera_centers(ig_scene_path):
    global ref_camera_center_list
    if len(ref_camera_center_list) > 0:
        return
    # read 50 cameras
    for i in range(50):
        scene_dict = read_json(os.path.join(ig_scene_path, f"{i:05d}/data.json"))
        ref_camera_center_list += [np.array(scene_dict["camera"]["pos"])]


def get_rand_camera_pose_from_ref():
    idx_1 = random.randint(0, len(ref_camera_center_list) - 1)
    idx_2 = random.randint(0, len(ref_camera_center_list) - 1)
    interp = np.random.rand()
    # import ipdb

    # ipdb.set_trace()
    # print(idx_1, idx_2)
    rand_center = ref_camera_center_list[idx_1] * interp + ref_camera_center_list[
        idx_2
    ] * (1 - interp)
    # yaw_angle = np.random.rand() * 360
    yaw_angle = np.random.randint(4) * 90
    rot_base = R.from_euler("x", -90, degrees=True).as_matrix()
    rotation = R.from_euler("z", yaw_angle, degrees=True).as_matrix() @ rot_base
    Twc = np.eye(4)
    Twc[:3, :3] = rotation
    Twc[:3, 3] = rand_center
    return Twc


def generate_rand_from_full_arrangement(visualizer, ig_scene_path, output_json_path):
    scene_dict = read_json(os.path.join(ig_scene_path, "full/data.json"))
    scene_dict_out = {}
    scene_dict_aug = copy.deepcopy(scene_dict)

    for idx, obj in enumerate(scene_dict_aug["objs"]):
        obj_type = obj["classname"]
        obj_id = obj["id"]
        bbox3d = obj["bdb3d"]

        length = np.array(bbox3d["size"]) * 0.5
        # bbox center
        center = np.array(bbox3d["centroid"])
        # position of object
        scale = np.array(bbox3d["scale"])
        rotation = np.array(bbox3d["basis"]).reshape(3, 3)

        # set augmentation parameter
        # aug_trans = 1.0
        aug_trans = 0.3
        if obj_type not in ["window", "door"]:
            base_dir = np.array([0, -1, 0])
            center += rotation @ base_dir * 0.2
            rand_center_offset = (np.random.rand((3)) - 0.5) * aug_trans
            if obj_type in not_attach_floor_type:
                center += rand_center_offset
            else:
                center[:2] += rand_center_offset[:2]
            rot_aug = R.from_euler(
                "z",
                # np.random.rand() * 360,
                np.random.rand() * 0,
                # (np.random.rand() - 0.5) * 30,
                degrees=True,
            ).as_matrix()
        rotation = rot_aug @ rotation

        # write back to obj
        bbox3d["centroid"] = center.tolist()
        bbox3d["basis"] = rotation.tolist()
        scene_dict_aug["objs"][idx]["bdb3d"] = bbox3d

    read_camera_centers(ig_scene_path)
    # augment camera pose
    try_cnt = 0
    while True:
        # find a camera pose not inside the object bbox
        rand_Twc = get_rand_camera_pose_from_ref()
        is_cam_in_obj = False
        for obj in scene_dict_aug["objs"]:
            bbox = obj["bdb3d"]
            obj_type = obj["classname"]
            Two = np.eye(4)
            Two[:3, :3] = np.array(bbox["basis"]).reshape(3, 3)
            Two[:3, 3] = np.array(bbox["centroid"])
            cam_center = rand_Twc[:3, 3]
            Tow = np.linalg.inv(Two)
            cam_in_obj = Tow[:3, :3] @ cam_center + Tow[:3, 3]
            # print(cam_in_obj, bbox["size"])
            # print(cam_center, cam_in_obj)
            # if ((np.abs(cam_in_obj) - np.array(bbox["size"]) / 2) < 0).sum() != 0:
            bbox_large = np.array(bbox["size"]) + np.array([0.2, 0.2, 0])
            cam_in_obj_abs = np.abs(cam_in_obj)
            bbox_large_half = bbox_large / 2
            if (
                cam_in_obj_abs[0] < bbox_large_half[0]
                and cam_in_obj_abs[1] < bbox_large_half[1]
            ):
                print("cam in obejct, resample camera pose", try_cnt)
                try_cnt += 1
                is_cam_in_obj = True
                break
            # bbox_large[2] = 1000
            # bbox_large *= 150
            # print()
            # ipdb.set_trace()
            # if ((np.abs(cam_in_obj) - bbox_large / 2) < 0).sum() > 0:
            #     print("cam in obejct, resample camera pose", try_cnt)
            #     try_cnt += 1
            #     cam_in_obj = True
            # break
            # if obj_type == "bed":
            #     print("cam in obejct, resample camera pose", try_cnt)
            #     try_cnt += 1
            #     cam_in_obj = True
            #     break
        if is_cam_in_obj is False:
            break
        # else:
        #     print("find camera in obj, next sample")

    # write camera
    camera_dict = scene_dict_aug["camera"]
    camera_dict["pos"] = rand_Twc[:3, 3].tolist()
    target = np.array([0, 0, 1])
    target = rand_Twc[:3, 3] + rand_Twc[:3, :3] @ target
    up = np.array([0, -1, 0])
    up = rand_Twc[:3, :3] @ target
    camera_dict["target"] = target.tolist()
    camera_dict["cam3d2world"] = rand_Twc.tolist()
    camera_dict["world2cam3d"] = np.linalg.inv(rand_Twc).tolist()

    # delete uncertain
    for obj_item in scene_dict_aug["objs"]:
        obj_item.pop("bfov", None)
        obj_item.pop("bdb2d", None)
        obj_item.pop("contour", None)
    write_json(scene_dict_aug, output_json_path)
    # draw_poses(visualizer, [rand_Twc])
    # write_json(scene_dict_aug, "/tmp/scene_aug.json")
    # ig_scene_path = "data/ig_dataset_v1.0.1"
    # draw_ig_scene(visualizer, "/tmp/scene_aug.json", ig_scene_path)
    return rand_Twc


if __name__ == "__main__":
    # params
    parser = argparse.ArgumentParser()
    # data paths
    parser.add_argument("--pcd", default=None)
    parser.add_argument("--mesh", default=None)
    parser.add_argument("--transform_json", default=None)
    parser.add_argument("--ig_scene_json", default=None)
    parser.add_argument("--ig_scene_path", default=None)
    parser.add_argument("--real_scene_json", default=None)
    args = parser.parse_args()

    visualizer = O3dVisualizer()
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1.0, origin=[0, 0, 0]
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
        draw_poses_from_transform_json(visualizer, read_json(args.transform_json))

    # poses = [
    #     np.array(
    #         [
    #             [1.0, 0.0, 0.0, -5.099999904632568],
    #             [0.0, 0.0, 1.0, -3.200000047683716],
    #             [-0.0, -1.0, -0.0, 1.600000023841858],
    #             [0.0, 0.0, 0.0, 1.0],
    #         ]
    #     ).reshape(4, 4),
    # ]
    # # import ipdb; ipdb.set_trace()
    # draw_poses(visualizer, poses)

    if args.ig_scene_json:
        # Nx(cx, cy, cz, dx, dy, dz, semantic_label)
        ig_scene_path = "data/ig_dataset_v1.0.1"
        draw_ig_scene(visualizer, args.ig_scene_json, ig_scene_path)

    num_random = 1000

    out_dir = "debug/ig_scene_rand_arrangement/"
    poses = []
    for i in tqdm(range(num_random)):
        out_subfolder = os.path.join(out_dir, f"{i:05d}")
        os.makedirs(out_subfolder, exist_ok=True)
        Twc = generate_rand_from_full_arrangement(
            visualizer, args.ig_scene_path, os.path.join(out_subfolder, "data.json")
        )
        poses += [Twc]
    draw_poses(visualizer, poses)

    visualizer.run_visualize()
