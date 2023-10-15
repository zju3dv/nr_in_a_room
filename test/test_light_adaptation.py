import sys
import os

os.environ["OMP_NUM_THREADS"] = "1"  # noqa
os.environ["MKL_NUM_THREADS"] = "1"  # noqa
sys.path.append(".")  # noqa

import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from optim.room_optimizer import RoomOptimizer
from optim.misc_utils import (
    read_real_scene_localization,
    read_real_scene_localization_with_name,
    read_testing_config,
)
from utils.util import read_json


def main(config):
    # active_instance_id = config.active_instance_id
    scene_info_json_path = config.scene_info_json
    active_instance_id = []
    for obj_info in read_json(scene_info_json_path)["objs"]:
        active_instance_id += [obj_info["id"]]
    if 0 not in active_instance_id:
        active_instance_id += [0]

    active_instance_id = [0, 33, 35, 37]
    image_path = config.test_image_path
    dataset_config = config.dataset_config["dataset"]

    bg_scale_factor = 1
    bg_scene_center = [0, 0, 0]
    if config.bg_dataset_config != "":
        bg_dataset_config = config.bg_dataset_config["dataset"]
        bg_scale_factor = bg_dataset_config["scale_factor"]
        bg_scene_center = bg_dataset_config["scene_center"]

    img_wh = config.img_wh
    # read image
    input_rgb = Image.open(image_path)
    input_rgb = input_rgb.resize(img_wh, Image.LANCZOS)
    input_rgb = np.array(input_rgb)
    input_rgb = torch.from_numpy(input_rgb).float() / 255  # (H, W, 3)

    refine_pose = config.get("refine_pose", False)

    print("refine_pose = ", refine_pose)

    # intialize room optimizer
    room_optimizer = RoomOptimizer(
        scene_info_json_path=config.scene_info_json,
        scale_factor=dataset_config["scale_factor"],
        scale_factor_dict=dataset_config.get("scale_factor_dict", {}),
        bg_scale_factor=bg_scale_factor,
        bg_scene_center=bg_scene_center,
        img_wh=config.img_wh,
        near=0.3,
        far=10.0,
        N_samples=64,
        N_importance=128,
        chunk=config.chunk,
        model_ckpt_path_dict=config.ckpt_path_dict,
        # relation_info=relation_info,
        relation_info={},
        output_path="debug",
        prefix=config.prefix,
        active_instance_id=active_instance_id,
        lr=1e-2,
        # lr=5e-2,
        N_optim_step=500,
        adjust_lr_per_step=0,
        optim_batch_size=1024,
        # optim_batch_size=2048,
        # optim_batch_size=4096,
        # use_amp=False,
        use_amp=True,
        optimize_light_env=True,
        # use_light_from_image_attr=True,
        optimize_appearance_code=config.get("optimize_appearance_code", False),
        mask_per_object=False,
        bbox_ray_intersect=True,
        bbox_enlarge=0.1,
        optimize_option=[
            "keypoint_mask",
            "photometric_loss",
            "" if refine_pose else "fix_object_pose",
            # "perceptual_loss",
            "z_axis_align_loss" if refine_pose else "",
            "object_room_wall_attach" if refine_pose else "",
            "object_room_floor_attach" if refine_pose else "",
            "physical_violation" if refine_pose else "",
            "physical_violation_delayed_start" if refine_pose else "",
            # "object_object_attach",
            # "viewing_constraint",
            "optimize_exposure",
            # "visualize_pred",
            # "print_loss_dict",
        ],
    )

    # room_optimizer.set_sampling_mask_from_seg(
    #     seg_mask=None,
    #     seg_mask_path=config.seg_mask_path,
    #     # add_noise_to_seg=0,
    #     add_noise_to_seg=5,  # dilate mask
    #     convert_seg_mask_to_box_mask=True,
    #     # convert_seg_mask_to_box_mask=False,
    # )

    room_optimizer.set_sampling_mask_from_seg(
        seg_mask=None,
        seg_mask_path=None,
        # add_noise_to_seg=0,
        add_noise_to_seg=5,  # dilate mask
        convert_seg_mask_to_box_mask=False,
        # convert_seg_mask_to_box_mask=False,
    )

    if "obj_prediction_json" in config:
        room_optimizer.set_initial_pose_from_prediction(config["obj_prediction_json"])
    else:
        room_optimizer.set_initial_object_poses_from_scene_meta(add_noise=False)

    # dump config
    config["optimize_option"] = room_optimizer.optimize_option
    OmegaConf.save(
        config=config,
        f=os.path.join(room_optimizer.output_path, "optim_config_full.yaml"),
    )

    room_optimizer.generate_relation()

    arrangement_name = config.arrangement_name
    real_room_loc = read_real_scene_localization_with_name(arrangement_name)
    # get file stem
    stem = os.path.splitext(os.path.basename(image_path))[0]
    pose = real_room_loc[stem]["pose_slam_Twc"]
    pose = np.array(pose)

    room_optimizer.optimize(
        input_rgb,
        pose=pose,
    )


if __name__ == "__main__":
    """
    Usage:
        python test/test_light_adaptation.py  config=test/config/ig_bedroom.yml "img_wh=[320,180]" prefix=dbg_bedroom_light
    """
    config = read_testing_config()
    main(config)
