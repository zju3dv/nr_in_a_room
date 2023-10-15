from posix import listdir
import sys
import os

os.environ["OMP_NUM_THREADS"] = "1"  # noqa
os.environ["MKL_NUM_THREADS"] = "1"  # noqa
sys.path.append(".")  # noqa

import torch
import numpy as np
import os.path as osp
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf
from optim.room_optimizer import RoomOptimizer
from optim.misc_utils import read_testing_config
from utils.util import list_dir, read_json


def prepare_room_optimizer(config, scene_info_json_path):
    active_instance_id = config.active_instance_id

    dataset_config = config.dataset_config["dataset"]

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
        # optim_batch_size=1024,
        optim_batch_size=2048,
        # optim_batch_size=4096,
        # use_amp=False,
        use_amp=True,
        optimize_light_env=True,
        use_light_from_image_attr=True,
        optimize_appearance_code=config.get("optimize_appearance_code", False),
        mask_per_object=False,
        bbox_ray_intersect=True,
        bbox_enlarge=0.1,
        optimize_option=[
            "keypoint_mask",
            "photometric_loss",
            # "perceptual_loss",
            "z_axis_align_loss",
            "object_room_wall_attach",
            "object_room_floor_attach",
            "physical_violation",
            # "physical_violation_delayed_start",
            "object_object_attach",
            "viewing_constraint",
            # "optimize_exposure",
            "regenerate_relation_during_test",
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

    # if "obj_prediction_json" in config:
    #     room_optimizer.set_initial_pose_from_prediction(config["obj_prediction_json"])
    # else:
    #     room_optimizer.set_initial_object_poses_from_scene_meta()

    # room_optimizer.generate_relation()

    # room_optimizer.optimize(input_rgb)
    return room_optimizer


def main(config):
    scene_name = config["scene_name"]
    pred_src_dir = config["pred_src_dir"]

    pred_src_scene_dir = osp.join("data/object_prediction", pred_src_dir, scene_name)
    print("Using prediction from", pred_src_scene_dir)

    # currently only support multi_lights
    multi_case_dirs = list_dir(pred_src_scene_dir)
    print("Find cases", multi_case_dirs)

    # prepare room optimizer and backup config
    room_optimizer = prepare_room_optimizer(
        config,
        f"data/{scene_name}/scene/full/data.json",  # also load scene info json,
        # which can be an empty placeholder in the future
    )
    config["optimize_option"] = room_optimizer.optimize_option
    OmegaConf.save(
        config=config,
        f=os.path.join(room_optimizer.output_path, "optim_config_full.json"),
    )

    output_path_base = room_optimizer.output_path

    for curr_case in tqdm(multi_case_dirs):
        items = list_dir(osp.join(pred_src_scene_dir, curr_case))
        for item in tqdm(items):
            item_dir = osp.join(pred_src_scene_dir, curr_case, item)
            print("Working on", item_dir)

            pred_json = osp.join(item_dir, "pred.json")

            active_instance_id = list(map(int, list(read_json(pred_json).keys())))
            active_instance_id = [x for x in active_instance_id if x > 0]
            active_instance_id += [0]

            # reset optimizer state
            room_optimizer.reset_active_instance_id(active_instance_id)
            room_optimizer.reset_optimizable_parameters()
            room_optimizer.set_output_path(
                output_path_base, f"{curr_case}/{item}", with_timestamp=False
            )

            src_item_dir = (
                # f"data/{scene_name}/scene_multi_lights/{curr_case}/{item}/"
                f"data/{scene_name}/scene_custom_arrange_multi_lights/{curr_case}/{item}/"
            )
            # TODO: seg should be provided by object detector
            # seg_mask_path = f"{src_item_dir}/seg.png"
            seg_mask_path = osp.join(item_dir, "seg.png")
            room_optimizer.set_sampling_mask_from_seg(
                seg_mask_path=seg_mask_path,
                add_noise_to_seg=5,  # dilate mask
                convert_seg_mask_to_box_mask=False,
            )
            room_optimizer.set_initial_pose_from_prediction(pred_json)
            room_optimizer.generate_relation()

            image_path = f"{src_item_dir}/rgb.png"
            print(image_path)

            img_wh = config.img_wh
            # read image
            input_rgb = Image.open(image_path)
            input_rgb = input_rgb.resize(img_wh, Image.LANCZOS)
            input_rgb = np.array(input_rgb)
            input_rgb = torch.from_numpy(input_rgb).float() / 255  # (H, W, 3)

            pose = np.array(
                read_json(f"{src_item_dir}/data.json")["camera"]["cam3d2world"]
            ).reshape(4, 4)

            room_optimizer.optimize(input_rgb, pose=pose)


if __name__ == "__main__":
    """
    Usage:
        python test/batch_optim_pano.py  config=test/config/ig_bedroom.yml "img_wh=[320,180]" pred_src_dir=DeepPano_wo_relation prefix=dbg_bedroom_batch
    """
    config = read_testing_config()
    main(config)
