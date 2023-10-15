# ref: https://github.com/StanfordVL/iGibson/blob/master/igibson/utils/data_utils/ext_object/scripts/step_5_visualizations.py
import sys
import argparse
import json
import os
import subprocess
from shutil import which

import numpy as np
from PIL import Image
from transforms3d.euler import euler2quat

import igibson
from igibson.objects.articulated_object import ArticulatedObject, URDFObject
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.render.profiler import Profiler
from igibson.scenes.empty_scene import EmptyScene
from igibson.simulator import Simulator
from igibson.utils.utils import quatToXYZW, rotate_vector_2d

import cv2

sys.path.append(".")
from utils.util import read_json, write_json
from data_gen.data_geo_utils import create_sphere_lookat_poses

image_width = 1920
image_height = 1440
camera_angle_x = 80
# radius_factor = 1.5
# min_radius = 1.5
# radius_plus = 1.0
radius_plus = 0.5

parser = argparse.ArgumentParser("Generate visulization for iGibson object")
parser.add_argument("--ig_scene_json", dest="ig_scene_json")
parser.add_argument("--num_views", dest="num_views", default=50, type=int)
parser.add_argument("--hdr_texture_file", default="default_texture", type=str)
parser.add_argument("--output_prefix", default="", type=str)


def render_single_object(
    obj_meta_info, model_path, output_dir, num_views, hdr_texture_file
):
    os.makedirs(output_dir, exist_ok=True)
    path_list = model_path.split("/")
    model_id = path_list[-1]
    category = path_list[-2]

    # import ipdb; ipdb.set_trace()
    # hdr_texture = os.path.join(
    #     igibson.ig_dataset_path, 'scenes', 'background', 'probe_02.hdr')
    # hdr_texture2 = os.path.join(
    #     igibson.ig_dataset_path, 'scenes', 'background', 'probe_03.hdr')
    if hdr_texture_file == "default_texture":
        hdr_texture = os.path.join(
            igibson.ig_dataset_path, "scenes", "background", "probe_03.hdr"
        )
        hdr_texture2 = os.path.join(
            igibson.ig_dataset_path, "scenes", "background", "probe_03.hdr"
        )
    else:
        hdr_texture = hdr_texture_file
        hdr_texture2 = hdr_texture_file
    # hdr_texture = os.path.join(igibson.ig_dataset_path, "scenes", "background",
    #                            "photo_studio_01_2k.hdr")
    # hdr_texture2 = os.path.join(igibson.ig_dataset_path, "scenes", "background",
    #                            "photo_studio_01_2k.hdr")
    # settings = MeshRendererSettings(env_texture_filename=hdr_texture, enable_shadow=True, msaa=True)
    settings = MeshRendererSettings(
        env_texture_filename=hdr_texture,
        env_texture_filename2=hdr_texture2,
        # env_texture_filename3=hdr_texture,
        enable_shadow=True,
        msaa=False,
    )
    # compute v_fov from h_fov (camera angle x)
    focal = (image_width / 2) / np.tan((camera_angle_x / 2) / (180 / np.pi))
    v_fov = np.arctan((image_height / 2) / focal) * 2 * 180 / np.pi

    s = Simulator(
        mode="headless",
        image_width=image_width,
        image_height=image_height,
        vertical_fov=v_fov,
        rendering_settings=settings,
    )
    scene = EmptyScene()
    s.import_scene(scene, render_floor_plane=False)

    s.renderer.set_light_position_direction([0, 0, 10], [0, 0, 0])

    ###########################
    # Get center and scale
    ###########################
    bbox_json = os.path.join(model_path, "misc", "metadata.json")
    with open(bbox_json, "r") as fp:
        bbox_data = json.load(fp)
        # scale = 1.5 / max(bbox_data["bbox_size"])
        scale = np.array(obj_meta_info["bdb3d"]["scale"])
        max_bbox_size = np.linalg.norm(np.array(obj_meta_info["bdb3d"]["size"]))
        # center = -scale * np.array(bbox_data["base_link_offset"])
        center = -np.multiply(scale, np.array(bbox_data["base_link_offset"]))

    urdf_path = os.path.join(model_path, "{}.urdf".format(model_id))
    print(urdf_path)
    print("current_id", obj_meta_info["id"])

    obj = ArticulatedObject(filename=urdf_path)
    s.import_object(obj, custom_scale=scale, class_id=obj_meta_info["id"])
    # recenter object
    obj.set_position(center)

    save_dir = os.path.join(output_dir, "full")
    os.makedirs(save_dir, exist_ok=True)
    radius = max_bbox_size + radius_plus
    poses, eyes = create_sphere_lookat_poses(
        radius=radius,
        n_poses=num_views,
        n_circles=3,
        up_dir="z",
        phi_begin=20,
        phi_end=90,
    )

    for i in range(num_views):
        s.renderer.set_camera(eyes[i], [0, 0, 0], [0, 0, 1])
        s.sync()
        frame = s.renderer.render(modes=("rgb", "normal", "seg", "3d"))
        rgb, normal, seg, scn_3d = frame

        # save rgb
        img = Image.fromarray((255 * rgb[:, :, :3]).astype(np.uint8))
        img.save(os.path.join(save_dir, "{:05d}.png".format(i)))

        # save normal
        img = Image.fromarray((255 * normal[:, :, :3]).astype(np.uint8))
        img.save(os.path.join(save_dir, "{:05d}.normal.png".format(i)))

        # import ipdb; ipdb.set_trace()
        # save seg
        seg = np.round((seg[:, :, 0] * 255))
        cv2.imwrite(
            os.path.join(save_dir, "{:05d}.seg.png".format(i)), seg.astype(np.uint16)
        )

        # save depth
        depth = scn_3d[:, :, 2] * -1
        cv2.imwrite(
            os.path.join(save_dir, "{:05d}.depth.png".format(i)),
            (depth * 1000).astype(np.uint16),
        )
        np.save(
            os.path.join(save_dir, "{:05d}.depth.npy".format(i)),
            (depth),
        )

    s.disconnect()
    return poses


def main():
    args = parser.parse_args()
    ig_scene_info = read_json(args.ig_scene_json)
    num_views = args.num_views
    hdr_texture_file = args.hdr_texture_file
    base_output_dir = f"debug/generate/"
    ig_data_base_dir = "data/ig_dataset_v1.0.1"
    scene_name = ig_scene_info["scene"]
    room_name = ig_scene_info["room"]

    os.makedirs(base_output_dir, exist_ok=True)
    output_object_dir = (
        "objects"
        if args.output_prefix == ""
        else f"objects_multi_lights/{args.output_prefix}"
    )
    for idx_b, obj in enumerate(ig_scene_info["objs"]):
        model_path = os.path.join(
            ig_data_base_dir, "objects", obj["model_path"]
        ).strip()
        output_dir = os.path.join(
            base_output_dir,
            scene_name + "_" + room_name,
            output_object_dir,
            "{:03d}".format(obj["id"]),
            obj["model_path"],
        ).strip()
        poses = render_single_object(
            obj, model_path, output_dir, num_views, hdr_texture_file
        )

        transforms_info = {
            "camera_angle_x": camera_angle_x / (180 / np.pi),
            "frames": [],
        }
        for idx, pose in enumerate(poses):
            transforms_info["frames"].append(
                {
                    "idx": idx,
                    "transform_matrix": pose.tolist(),
                    "file_path": f"./full/{idx:05d}",
                }
            )

        write_json(transforms_info, os.path.join(output_dir, "transforms_full.json"))


if __name__ == "__main__":
    main()
