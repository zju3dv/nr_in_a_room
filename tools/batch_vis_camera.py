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
from utils.util import map_to_color, read_json
from scipy.spatial.transform import Rotation as R


def draw_poses(visualizer, poses, axis_size=0.1):
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
        axis_size = axis_size
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
    visualizer.add_line_set(lines_pt, lines_idx, colors=lines_color, radius=0.01)

    camera_centers = np.array(camera_centers)
    if len(poses) > 1:
        visualizer.add_np_points(
            camera_centers,
            color=map_to_color(np.arange(0, len(poses)), cmap="plasma"),
            size=0.01,
        )


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
        print(id_str)

        length = np.array(bbox3d["size"]) * 0.5
        # bbox center
        center = np.array(bbox3d["centroid"])
        # position of object
        scale = np.array(bbox3d["scale"])
        rotation = np.array(bbox3d["basis"]).reshape(3, 3)

        Two = np.eye(4)
        Two[:3, :3] = rotation
        Two[:3, 3] = center
        draw_poses(visualizer, [Two], 0.8)

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
    draw_poses(
        visualizer, [np.array(ig_scene_info["camera"]["cam3d2world"]).reshape(4, 4)]
    )


if __name__ == "__main__":
    # params
    parser = argparse.ArgumentParser()
    # data paths
    parser.add_argument("--pcd", default=None)
    parser.add_argument("--mesh", default=None)
    parser.add_argument("--ig_scene_json", default=None)
    parser.add_argument("--ig_scene_path", default=None)
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

    poses = []
    for i in range(50):
        scene_info_json = os.path.join(args.ig_scene_path, f"{i:05d}/data.json")
        Twc = read_json(scene_info_json)["camera"]["cam3d2world"]
        poses += [np.array(Twc).reshape(4, 4)]
    draw_poses(visualizer, poses)
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

    visualizer.run_visualize()
