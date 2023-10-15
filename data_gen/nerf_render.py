import sys
import os

os.environ["OMP_NUM_THREADS"] = "1"  # noqa
os.environ["MKL_NUM_THREADS"] = "1"  # noqa
sys.path.append(".")  # noqa

import torch
import numpy as np
import imageio
import time
from argparse import ArgumentParser
from utils.util import read_json, read_yaml
from optim.room_optimizer import RoomOptimizer
from optim.misc_utils import read_testing_config


def get_id_list(scene_meta):
    id_list = []
    for o in scene_meta["objs"]:
        id_list.append(o["id"])
    return id_list


def intersection(scene_meta, active_id_list):
    scene_id_list = get_id_list(scene_meta)
    has_background = 0 in active_id_list
    inter = [value for value in scene_id_list if value in active_id_list]
    if has_background:
        inter += [0]
    return inter


def render_image(fixed_params, scene_json, save_path):
    dataset_config, bg_scale_factor, bg_scene_center, active_instance_id = fixed_params
    scene_meta = read_json(scene_json)
    active_instance_id = intersection(scene_meta, active_instance_id)
    # intialize room optimizer
    room_optimizer = RoomOptimizer(
        scene_info_json_path=scene_json,
        scale_factor=dataset_config["scale_factor"],
        scale_factor_dict=dataset_config.get("scale_factor_dict", {}),
        bg_scale_factor=bg_scale_factor,
        bg_scene_center=bg_scene_center,
        img_wh=config.img_wh,
        near=0.3,
        far=10.0,
        chunk=config.chunk,
        model_ckpt_path_dict=config.ckpt_path_dict,
        active_instance_id=active_instance_id,
        use_amp=True,
        use_light_from_image_attr=True,  # we use fixed light code (e.g. probe_03)
        optimize_appearance_code=config.get("optimize_appearance_code", False),
        use_appearance_from_image_attr=True,
    )

    # initialize object poses with no noise
    room_optimizer.set_initial_object_poses_from_scene_meta(add_noise=False)

    # we show an example to use pose
    pose = np.array(scene_meta["camera"]["cam3d2world"]).reshape(4, 4)
    # Original poses has rotation in form "right down forward", change to NDC "right up back"
    fix_rot = np.array([1, 0, 0, 0, -1, 0, 0, 0, -1]).reshape(3, 3)
    pose[:3, :3] = pose[:3, :3] @ fix_rot

    t1 = time.time()
    image_np = room_optimizer.render_full_scene(
        pose=pose,
        idx=-1,
        return_raw_image=True,
    )
    t2 = time.time()

    print(f"Rendering finish in {t2-t1} s.")

    os.makedirs("debug", exist_ok=True)
    imageio.imwrite(save_path, image_np)


def main(config):
    # set instance id to visualize
    # active_instance_id = [48, 4, 9, 104]
    active_instance_id = [3, 4, 5, 6, 8, 9, 10, 48, 49, 50, 51, 62, 64, 94, 97, 104]
    # active_instance_id = [46, 4, 9, 102]
    # also add background to rendering, background with instance_id = 0
    active_instance_id += [0]

    dataset_config = config.dataset_config["dataset"]

    bg_scale_factor = 1
    bg_scene_center = [0, 0, 0]
    if config.bg_dataset_config != "":
        bg_dataset_config = config.bg_dataset_config["dataset"]
        bg_scale_factor = bg_dataset_config["scale_factor"]
        bg_scene_center = bg_dataset_config["scene_center"]

    fixed_params = dataset_config, bg_scale_factor, bg_scene_center, active_instance_id
    train_json = read_json(f"{config.workspace}/train.json")
    for item in train_json:
        jpath = f"{config.workspace}/{item}".replace("data.pkl", "data.json")
        spath = f"{config.workspace}/{item}".replace("data.pkl", "nerf.png")
        render_image(fixed_params, jpath, spath)


if __name__ == "__main__":
    """
    Example:
    python data_gen/nerf_render.py \
        config=test/config/ig_bedroom.yml \
        "img_wh=[1024,512] \
        workspace=data/igibson/
    """
    config = read_testing_config()
    main(config)
