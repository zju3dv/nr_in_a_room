# ref: https://github.com/StanfordVL/iGibson/blob/master/igibson/utils/data_utils/ext_object/scripts/step_5_visualizations.py

import argparse
import ipaddress
import json
import os
import subprocess
from shutil import which

import numpy as np
from PIL import Image
from transforms3d.euler import euler2quat

import igibson
from igibson.objects.articulated_object import ArticulatedObject
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.render.profiler import Profiler
from igibson.scenes.empty_scene import EmptyScene
from igibson.simulator import Simulator
from igibson.utils.utils import quatToXYZW, rotate_vector_2d

import cv2

parser = argparse.ArgumentParser("Generate visulization for iGibson object")
parser.add_argument("--input_dir", dest="input_dir")


def main():
    global _mouse_ix, _mouse_iy, down, view_direction

    args = parser.parse_args()
    model_path = args.input_dir
    print(model_path)

    model_id = os.path.basename(model_path)
    category = os.path.basename(os.path.dirname(model_path))

    hdr_texture = os.path.join(
        igibson.ig_dataset_path, "scenes", "background", "probe_03.hdr"
    )
    # settings = MeshRendererSettings(env_texture_filename=hdr_texture, enable_shadow=True, msaa=True)
    settings = MeshRendererSettings(
        env_texture_filename=hdr_texture,
        enable_shadow=True,
        msaa=False,
    )

    s = Simulator(
        mode="headless",
        image_width=1800,
        image_height=1200,
        vertical_fov=70,
        rendering_settings=settings,
    )
    scene = EmptyScene()
    s.import_scene(scene, render_floor_plane=False)

    s.renderer.set_light_position_direction([0, 0, 10], [0, 0, 0])

    # s.renderer.load_object("plane/plane_z_up_0.obj", scale=[3, 3, 3])
    # s.renderer.load_object("/home/ybbbbt/Developer/iGibson/igibson/utils/data_utils/ext_object/scripts/plane/plane_z_up_0.obj", scale=[3, 3, 3])
    # s.renderer.add_instance(0)
    # s.renderer.set_pose([0, 0, -1.5, 1, 0, 0.0, 0.0], -1)

    ###########################
    # Get center and scale
    ###########################
    bbox_json = os.path.join(model_path, "misc", "metadata.json")
    with open(bbox_json, "r") as fp:
        bbox_data = json.load(fp)
        scale = 1.5 / max(bbox_data["bbox_size"])
        center = -scale * np.array(bbox_data["base_link_offset"])

    urdf_path = os.path.join(model_path, "{}.urdf".format(model_id))
    print(urdf_path)
    obj = ArticulatedObject(filename=urdf_path, scale=scale)
    s.import_object(obj)

    _mouse_ix, _mouse_iy = -1, -1
    down = False

    theta, r = 0, 1.5

    px = r * np.sin(theta)
    py = r * np.cos(theta)
    pz = 1
    camera_pose = np.array([px, py, pz])
    s.renderer.set_camera(camera_pose, [0, 0, 0], [0, 0, 1])

    # import ipdb; ipdb.set_trace()

    num_views = 6
    save_dir = os.path.join("debug", "visualizations")
    os.makedirs(save_dir, exist_ok=True)
    for i in range(num_views):
        theta = np.pi * 2 / num_views * i
        pos = np.append(rotate_vector_2d(center[:2], theta), center[2])
        orn = quatToXYZW(euler2quat(0.0, 0.0, -theta), "wxyz")
        obj.set_position_orientation(pos, orn)
        s.sync()
        # with Profiler("Render"):
        # frame = s.renderer.render(modes=("rgb"))
        frame = s.renderer.render(modes=("rgb", "normal", "seg", "3d"))
        # import ipdb; ipdb.set_trace()
        rgb, normal, seg, scn_3d = frame

        # img = Image.fromarray((255 * np.concatenate(frame, axis=1)[:, :, :3]).astype(np.uint8))
        # img.save(os.path.join(save_dir, "{:02d}.png".format(i)))

        # save rgb
        img = Image.fromarray((255 * rgb[:, :, :3]).astype(np.uint8))
        img.save(os.path.join(save_dir, "{:02d}.png".format(i)))

        # save normal
        img = Image.fromarray((255 * normal[:, :, :3]).astype(np.uint8))
        img.save(os.path.join(save_dir, "{:02d}.normal.png".format(i)))

        # import ipdb; ipdb.set_trace()
        # save seg
        seg = np.round((seg[:, :, 0] * 255))
        cv2.imwrite(
            os.path.join(save_dir, "{:02d}.seg.png".format(i)), seg.astype(np.uint16)
        )

        # save depth
        depth = scn_3d[:, :, 2] * -1
        cv2.imwrite(
            os.path.join(save_dir, "{:02d}.depth.png".format(i)),
            (depth * 1000).astype(np.uint16),
        )

    # if which("ffmpeg") is not None:
    #     # cmd = "ffmpeg -framerate 2 -i {s}/%2d.png -y -r 16 -c:v libx264 -pix_fmt yuvj420p {s}/{m}.mp4".format(
    #     cmd = "/usr/bin/ffmpeg -framerate 2 -i {s}/%2d.png -y -r 16 -c:v libx264 -pix_fmt yuvj420p {s}/{m}.mp4".format(
    #         s=save_dir, m=model_id
    #     )
    #     subprocess.call(cmd, shell=True)
    # cmd = 'rm {}/*.png'.format(save_dir)
    # subprocess.call(cmd, shell=True)


if __name__ == "__main__":
    main()
