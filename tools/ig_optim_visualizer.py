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


def draw_ig_scene_with_custom_pose(
    visualizer: O3dVisualizer,
    ig_scene_json: str,
    ig_scene_path: str,
    optimized_meta_json=None,
):
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

    use_custom_pose = optimized_meta_json is not None
    if use_custom_pose:
        optimized_objs = read_json(optimized_meta_json)

    lines_pt, lines_idx, lines_color = [], [], []
    text_pts = []
    i = 0
    for idx_b, obj in enumerate(ig_scene_info["objs"]):
        obj_name = obj["model_path"].split("/")[-1]
        # get object bbox
        bbox3d = obj["bdb3d"]
        length = np.array(bbox3d["size"]) * 0.5
        scale = np.array(bbox3d["scale"])
        # get base_link_offset
        obj_meta_info = read_json(
            os.path.join(
                ig_scene_path, "objects", obj["model_path"], "misc/metadata.json"
            )
        )
        base_link_offset = np.array(obj_meta_info["base_link_offset"])

        if use_custom_pose:
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
            mesh.compute_vertex_normals()
            visualizer.add_o3d_geometry(mesh)

        # draw semantic id
        text_pos = center + np.array([0, 0, length[2]])
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

    assert len(lines_pt) > 0
    lines_pt = np.array(lines_pt)

    visualizer.add_line_set(lines_pt, lines_idx, colors=lines_color, radius=0.01)

    text_pts = np.concatenate(text_pts, axis=0)
    visualizer.add_np_points(text_pts, color=[1, 0, 0], size=0.01)


def draw_real_data_with_custom_pose(
    visualizer: O3dVisualizer,
    real_room_scene_json: str,
    base_data_path: str,
    optimized_meta_json=None,
):
    scene_info = read_json(real_room_scene_json)
    scene_name = scene_info["scene"]
    room_name = scene_info["room"]
    # add scene wall visualization
    dataset_path = os.path.join(base_data_path, f"{scene_name}_{room_name}")
    room_obj_path = os.path.join(dataset_path, "objects/000/background", "aligned.obj")

    room_mesh = o3d.io.read_triangle_mesh(room_obj_path, True)
    # room_mesh.compute_vertex_normals()
    visualizer.add_o3d_geometry(room_mesh)

    use_custom_pose = optimized_meta_json is not None
    if use_custom_pose:
        optimized_objs = read_json(optimized_meta_json)

    lines_pt, lines_idx, lines_color = [], [], []
    text_pts = []
    i = 0
    for idx_b, obj in enumerate(scene_info["objs"]):
        obj_id = obj["id"]
        if obj_id == 0:
            continue
        bbox3d = obj["bdb3d"]
        print(obj_id)

        length = np.array(bbox3d["size"]) * 0.5
        if use_custom_pose:
            obj_id = obj["id"]
            if str(obj_id) not in optimized_objs:
                continue
            Two = np.array(optimized_objs[str(obj_id)]["Two"]).reshape(4, 4)
            # Two = np.array(optimized_objs[str(obj_id)]["Tco"]).reshape(4, 4)
            rotation = Two[:3, :3]
            # center = Two[:3, 3] + rotation @ (base_link_offset * scale)
            center = Two[:3, 3]
        else:
            # bbox center
            center = np.array(bbox3d.get("centroid", np.zeros(3)))
            # position of object
            rotation = np.array(bbox3d.get("basis", np.eye(3))).reshape(3, 3)
        obj_filename = os.path.join(
            dataset_path, f"objects/{obj_id:03d}/{obj['model_path']}/aligned.obj"
        )

        mesh = o3d.io.read_triangle_mesh(obj_filename, True)
        # remove base link offset
        vertices = np.asarray(mesh.vertices)
        # scale object and transform to the correct place
        vertices = (rotation @ (vertices).T).T + center
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        # mesh.compute_vertex_normals()
        visualizer.add_o3d_geometry(mesh)

        # draw semantic id
        text_pos = np.array(center[0:2].tolist() + [0]) + np.array([0, 0, 3.0])
        curr_text_pts = visualizer.text_3d(
            str(obj_id),
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
    # draw_poses(
    #     visualizer, [np.array(scene_info["camera"]["cam3d2world"]).reshape(4, 4)]
    # )


if __name__ == "__main__":
    # params
    parser = argparse.ArgumentParser()
    # data paths
    parser.add_argument("--pcd", default=None)
    parser.add_argument("--mesh", default=None)
    parser.add_argument("--transform_json", default=None)
    parser.add_argument("--ig_scene_json", default=None)
    parser.add_argument("--ig_optim_json", default=None)
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

    if args.ig_scene_json:
        # Nx(cx, cy, cz, dx, dy, dz, semantic_label)
        ig_scene_path = "data/ig_dataset_v1.0.1"
        draw_ig_scene_with_custom_pose(
            visualizer, args.ig_scene_json, ig_scene_path, args.ig_optim_json
        )

    if args.real_scene_json:
        # Nx(cx, cy, cz, dx, dy, dz, semantic_label)
        draw_real_data_with_custom_pose(
            visualizer, args.real_scene_json, "data/", args.ig_optim_json
        )

    visualizer.run_visualize()
