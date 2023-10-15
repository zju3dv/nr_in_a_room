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
from utils.util import list_dir, map_to_color, read_json
from scipy.spatial.transform import Rotation as R

extracted_bbox_info = {}


def read_bounding_boxes(scene_info_dict):
    global extracted_bbox_info
    scene_name = scene_info_dict["scene"]
    dataset_name = ""
    if scene_name == "Beechwood_0_int":
        dataset_name = "Beechwood_0_int_lobby_0"
    elif scene_name == "Beechwood_1_int":
        dataset_name = "Beechwood_1_int_bedroom_0_no_random"
    elif scene_name == "Merom_1_int":
        dataset_name = "Merom_1_int_childs_room_0"
    elif scene_name == "Pomaria_0_int":
        dataset_name = "Pomaria_0_int_home_office_0"
    elif "real" in scene_name:
        dataset_name = "real_room_0"

    all_image_files = sorted(
        glob.glob(f"data/extracted_meshes/{dataset_name}/bbox_*.json")
    )
    # print(scene_name)
    # print(all_image_files, f"data/extracted_meshes/{dataset_name}/bbox_*.json")
    for file in all_image_files:
        base_file = os.path.basename(file)
        obj_id = int(base_file[5:-5])
        extracted_bbox_info[str(obj_id)] = np.array(read_json(file)["size"])
    # print(extracted_bbox_info)


def type2color(obj_type):
    color_dict = {
        "bed": [0.36, 0.54, 0.66],
        "bottom_cabinet": [1.0, 0.75, 0],
        "picture": [0.6, 0.4, 0.8],
        "mirror": [0.57, 0.36, 0.51],
        "floor_lamp": [0.9, 0.17, 0.31],
        "piano": [0.13, 0.67, 0.8],
        "chair": [1.0, 0.44, 0.37],
        "sofa_chair": [1.0, 0.44, 0.37],
        "stool": [1.0, 0.44, 0.37],
        "table": [0.19, 0.55, 0.91],
        "carpet": [0.5, 1.0, 0.83],
    }
    return color_dict.get(obj_type, [0, 1, 0])


def draw_ig_scene_with_custom_pose(
    visualizer: O3dVisualizer,
    ig_scene_json: str,
    ig_scene_path: str,
    optimized_meta_json=None,
):
    # connect pybullet
    cid = p.connect(p.DIRECT)

    ig_scene_info = read_json(ig_scene_json)
    read_bounding_boxes(ig_scene_info)
    scene_name = ig_scene_info["scene"]
    # add scene wall visualization
    wall_obj_path = os.path.join(
        ig_scene_path, "scenes", scene_name, "shape/visual", "wall_vm.obj"
    )
    wall_mesh = o3d.io.read_triangle_mesh(wall_obj_path, True)
    visualizer.add_o3d_geometry(wall_mesh)
    floor_num = 1
    if scene_name == "Beechwood_0_int":
        floor_num = 4
    for i_floor in range(floor_num):
        floor_obj_path = os.path.join(
            ig_scene_path,
            "scenes",
            scene_name,
            "shape/visual",
            f"floor_{i_floor}_vm.obj",
        )
        floor_mesh = o3d.io.read_triangle_mesh(floor_obj_path, True)
        visualizer.add_o3d_geometry(floor_mesh)

    use_custom_pose = optimized_meta_json is not None
    if use_custom_pose:
        optimized_objs = read_json(optimized_meta_json)

    lines_pt, lines_idx, lines_color = [], [], []
    text_pts = []
    i = 0
    for idx_b, obj in enumerate(ig_scene_info["objs"]):
        obj_name = obj["model_path"].split("/")[-1]
        obj_id = obj["id"]
        obj_type = obj["classname"]
        # get object bbox
        bbox3d = obj["bdb3d"]
        global extracted_bbox_info
        if str(obj_id) in extracted_bbox_info:
            length = extracted_bbox_info[str(obj_id)] * 0.5
            print(length)
        else:
            length = np.array(bbox3d["size"]) * 0.5
        scale = np.array(bbox3d["scale"])
        # get base_link_offset
        obj_meta_info = read_json(
            os.path.join(
                ig_scene_path, "objects", obj["model_path"], "misc/metadata.json"
            )
        )
        base_link_offset = np.array(obj_meta_info["base_link_offset"])

        if use_custom_pose and obj_type not in ["door", "window"]:
            obj_id = obj["id"]
            if str(obj_id) not in optimized_objs:
                continue
            Two = np.array(optimized_objs[str(obj_id)]["Two"]).reshape(4, 4)
            rotation = Two[:3, :3]
            # center = Two[:3, 3] + rotation @ (base_link_offset * scale)
            center = Two[:3, 3]
        else:
            # bbox center
            center = np.array(bbox3d["centroid"])
            rotation = np.array(bbox3d["basis"]).reshape(3, 3)

        id_str = str(obj["id"])
        print(id_str)
        # obj_path = os.path.join(ig_scene_path, 'objects', obj['model_path'],
        #                         'shape/visual')
        # parse urdf file to obtain origin and filename
        urdf_file = os.path.join(
            ig_scene_path, "objects", obj["model_path"], "{}.urdf".format(obj_name)
        )

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
            # mesh.compute_vertex_normals()
            visualizer.add_o3d_geometry(mesh)

        if obj_type not in ["door", "window"]:
            # draw semantic id
            text_pos = center + np.array([0, 0, length[2]])
            curr_text_pts = visualizer.text_3d(
                id_str,
                text_pos,
                font="/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf",
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
            # lines_color.extend([np.array([0.36, 0.54, 0.66]) for x in range(12)])
            lines_color.extend([np.array(type2color(obj_type)) for x in range(12)])
            i += 1

    assert len(lines_pt) > 0
    lines_pt = np.array(lines_pt)

    visualizer.add_line_set(lines_pt, lines_idx, colors=lines_color, radius=0.01)

    text_pts = np.concatenate(text_pts, axis=0)
    # visualizer.add_np_points(text_pts, color=[1, 0, 0], size=0.01)


if __name__ == "__main__":
    # params
    parser = argparse.ArgumentParser()
    # data paths
    parser.add_argument("--pcd", default=None)
    parser.add_argument("--mesh", default=None)
    parser.add_argument("--transform_json", default=None)
    parser.add_argument("--ig_scene_json", default=None)
    parser.add_argument("--ig_optim_json", default=None)
    args = parser.parse_args()

    visualizer = O3dVisualizer()
    # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    #     size=1.0, origin=[0, 0, 0]
    # )
    # visualizer.add_o3d_geometry(mesh_frame)

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

    if args.ig_scene_json:
        # Nx(cx, cy, cz, dx, dy, dz, semantic_label)
        ig_scene_path = "data/ig_dataset_v1.0.1"
        draw_ig_scene_with_custom_pose(
            visualizer, args.ig_scene_json, ig_scene_path, args.ig_optim_json
        )

    visualizer.run_visualize()
