import sys
import os

os.environ["OMP_NUM_THREADS"] = "1"  # noqa
os.environ["MKL_NUM_THREADS"] = "1"  # noqa
sys.path.append(".")  # noqa

import torch
import numpy as np
from tqdm import tqdm
import imageio
import time
import cv2
from argparse import ArgumentParser
from utils.util import read_json, read_yaml
from optim.room_optimizer import RoomOptimizer
from optim.misc_utils import read_real_scene_localization, read_testing_config
from scipy.spatial.transform import Rotation


def render_frame(config, target_dir):
    # or load from config
    active_instance_id = config.active_instance_id

    dataset_config = config.dataset_config["dataset"]

    scene_info_json_path = config.scene_info_json

    active_instance_id = [0]
    for obj_info in read_json(scene_info_json_path)["objs"]:
        active_instance_id += [obj_info["id"]]

    bg_scale_factor = 1
    bg_scene_center = [0, 0, 0]
    if config.bg_dataset_config != "":
        bg_dataset_config = config.bg_dataset_config["dataset"]
        bg_scale_factor = bg_dataset_config["scale_factor"]
        bg_scene_center = bg_dataset_config["scene_center"]

    # intialize room optimizer
    room_optimizer = RoomOptimizer(
        scene_info_json_path=scene_info_json_path,
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
    scene_meta = read_json(scene_info_json_path)
    # localization_info = read_real_scene_localization(
    #     "/mnt/nas_54/group/BBYang/neural_scene_capture_360/capture_1104/processed/arrangement_panorama_select/arrangement1/traj.txt",
    #     "data/real_room_0/objects/000/background_hloc_neus_normal_converge/transform_info.json",
    # )
    pose = np.array(scene_meta["camera"]["cam3d2world"]).reshape(4, 4)
    # Original poses has rotation in form "right down forward", change to NDC "right up back"
    fix_rot = np.array([1, 0, 0, 0, -1, 0, 0, 0, -1]).reshape(3, 3)
    pose[:3, :3] = pose[:3, :3] @ fix_rot
    # from scipy.spatial.transform import Rotation as R

    # print(pose)

    # rot_fix_loc = np.array([0, 1, 0, 1, 0, 0, 0, 0, -1]).reshape(3, 3)
    # pose[:3, :3] = pose[:3, :3] @ rot_fix_loc
    # pose[:3, :3] = rot_fix_loc @ pose[:3, :3]

    t1 = time.time()
    # image_np = room_optimizer.render_full_scene(
    #     pose=pose,
    #     idx=-1,
    #     return_raw_image=True,
    #     refine_edge=True,
    #     # use_sphere_tracing=False,
    #     use_sphere_tracing=True,
    # )
    image_np, mask_np = room_optimizer.render_full_scene(
        pose=pose,
        idx=-1,
        return_raw_image=True,
        refine_edge=False,
        # refine_edge=True,
        # use_sphere_tracing=False,
        use_sphere_tracing=True,
        render_mask=True,
    )
    t2 = time.time()

    print(f"Rendering finish in {t2-t1} s.")

    os.makedirs("debug", exist_ok=True)
    imageio.imwrite(f"{target_dir}/rgb.png", image_np)
    cv2.imwrite(f"{target_dir}/seg.png", mask_np)


if __name__ == "__main__":
    """
    Example:
    python test/test_neural_scene_renderer.py \
        config=test/config/ig_bedroom.yml \
        "img_wh=[1024,512]" \
        base_dir=data/real_room_rand_arrangement
    """
    config = read_testing_config()
    # base_dir = "data/real_room_rand_arrangement/"
    base_dir = config.base_dir

    for idx in tqdm(range(1000)):
        curr_dir = f"{base_dir}/{idx:05d}"
        config.scene_info_json = curr_dir + "/data.json"
        render_frame(config, target_dir=curr_dir)
