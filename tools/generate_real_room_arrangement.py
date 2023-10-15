import os
from re import S
import sys
import ipdb
import pyglet

sys.path.append(".")  # noqa
pyglet.options["shadow_window"] = False
import argparse
import numpy as np
import torch
import cv2
import open3d as o3d
from tqdm import tqdm
import trimesh
import copy
import random

from utils.util import read_json, write_json
from scipy.spatial.transform import Rotation as R

SEED = 123
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

layout_pts = np.array(
    [
        [[1.296, 1.118, -1.33672], [1.26414, -0.634662, -1.34781]],
        [[1.296, 1.118, -1.33672], [-1.51183, 1.13223, -1.35334]],
        [[-1.51183, 1.13223, -1.35334], [-1.54128, -1.07319, -1.35649]],
        [[-1.54128, -1.07319, -1.35649], [0.877291, -1.12058, -1.33714]],
    ]
)

layout_rot = [-90, 0, 90, 180]

base_direction = np.array([0, -1, 0])

floor_z = -1.37


BBOX_PTS = [
    np.array([-1, -1, -1]),
    np.array([1, -1, -1]),
    np.array([-1, 1, -1]),
    np.array([1, 1, -1]),
    np.array([-1, -1, 1]),
    np.array([1, -1, 1]),
    np.array([-1, 1, 1]),
    np.array([1, 1, 1]),
]
BBOX_PTS = np.stack(BBOX_PTS, axis=0)  # [8, 3]


def compute_bbox_pts(obj_item):
    length = torch.Tensor(obj_item["bdb3d"]["size"]).float() * 0.5
    bbox_pts = (torch.from_numpy(BBOX_PTS).float() * length.view(1, 3)).numpy()
    Rwo = np.array(obj_item["bdb3d"]["basis"]).reshape(3, 3)
    two = np.array(obj_item["bdb3d"]["centroid"])
    bbox_pts = (Rwo @ bbox_pts.T).T + two
    return bbox_pts


def is_collision(obj_item_a, obj_item_b, th=0.2):
    bbox_pts_a = compute_bbox_pts(obj_item_a)
    bbox_pts_b = compute_bbox_pts(obj_item_b)
    # separating axis theorem, assume axis-align
    for i_axis in range(3):
        a_max, a_min = bbox_pts_a[:, i_axis].max(), bbox_pts_a[:, i_axis].min()
        b_max, b_min = bbox_pts_b[:, i_axis].max(), bbox_pts_b[:, i_axis].min()
        if b_max > a_min and b_min - a_max > th:
            return False
        if a_max > b_min and a_min - b_max > th:
            return False
    return True


def try_place_object_attach_wall(obj_item, curr_objs, layout_line_idx, interp=0.5):
    # place_info = {}
    Two = np.eye(4)
    rotation = R.from_euler("z", layout_rot[layout_line_idx], degrees=True).as_matrix()
    # offset to avoid collison
    # bbox_size = []
    bbox_size = obj_item["bdb3d"]["size"]
    start_end = layout_pts[layout_line_idx]
    # avoid too close to margin
    interp_margin = (bbox_size[0]) / 2 / np.linalg.norm(start_end[0] - start_end[1])
    interp = interp * (1 - interp_margin * 2) + interp_margin
    center = start_end[0] * interp + start_end[1] * (1 - interp)
    # z up
    center[2] += bbox_size[2] / 2
    # z_offset = floor_z + bbox_size[2] / 2
    # Two[2, 3] += z_offset
    # out offset
    # ipdb.set_trace()
    center += rotation @ base_direction * bbox_size[1] / 2
    # bbox_corners = []
    Two[:3, 3] = center
    Two[:3, :3] = rotation

    obj_item["bdb3d"]["centroid"] = Two[:3, 3].tolist()
    obj_item["bdb3d"]["basis"] = Two[:3, :3].tolist()

    for existing_obj in curr_objs:
        if is_collision(obj_item, existing_obj):
            return False
    return True


def try_place_object_center(
    obj_item, curr_objs, layout_line_idx, interp_x=0.5, interp_y=0.5
):
    # place_info = {}
    layout_line_idx = 1
    Two = np.eye(4)
    rotation = R.from_euler("z", layout_rot[layout_line_idx], degrees=True).as_matrix()
    # offset to avoid collison
    # bbox_size = []
    bbox_size = obj_item["bdb3d"]["size"]
    start_end_x = layout_pts[1]
    start_end_y = layout_pts[0]
    # avoid too close to margin
    interp_margin_x = (
        (bbox_size[0]) / 2 / np.linalg.norm(start_end_x[0] - start_end_x[1])
    )
    interp_margin_y = (
        (bbox_size[1]) / 2 / np.linalg.norm(start_end_y[0] - start_end_y[1])
    )
    interp_x = interp_x * (1 - interp_margin_x * 2) + interp_margin_x
    interp_y = interp_y * (1 - interp_margin_y * 2) + interp_margin_y
    center_x = start_end_x[0] * interp_x + start_end_x[1] * (1 - interp_x)
    center_y = start_end_y[0] * interp_y + start_end_y[1] * (1 - interp_y)
    center = center_x
    center[1] = center_y[1]
    # z up
    center[2] += bbox_size[2] / 2
    # z_offset = floor_z + bbox_size[2] / 2
    # Two[2, 3] += z_offset
    # out offset
    # ipdb.set_trace()
    center += rotation @ base_direction * bbox_size[1] / 2
    # bbox_corners = []
    Two[:3, 3] = center
    Two[:3, :3] = rotation

    obj_item["bdb3d"]["centroid"] = Two[:3, 3].tolist()
    obj_item["bdb3d"]["basis"] = Two[:3, :3].tolist()

    for existing_obj in curr_objs:
        if is_collision(obj_item, existing_obj):
            return False
    return True


def random_camera_in_room(interp_x, interp_y, yaw_angle, height=1.6):
    Twc = np.eye(4)
    # rot_base = np.array([0, 0, 1, 0, 1, 0, -1, 0, 0]).reshape(3, 3)
    rot_base = R.from_euler("x", -90, degrees=True).as_matrix()
    rotation = R.from_euler("z", yaw_angle, degrees=True).as_matrix() @ rot_base
    # offset to avoid collison
    # bbox_size = []
    start_end_x = layout_pts[1]
    start_end_y = layout_pts[0]
    # avoid too close to margin
    center_x = start_end_x[0] * interp_x + start_end_x[1] * (1 - interp_x)
    center_y = start_end_y[0] * interp_y + start_end_y[1] * (1 - interp_y)
    center = center_x
    center[1] = center_y[1]
    # z up
    center[2] += height
    # z_offset = floor_z + bbox_size[2] / 2
    # Two[2, 3] += z_offset
    # out offset
    # ipdb.set_trace()
    # bbox_corners = []
    Twc[:3, 3] = center
    Twc[:3, :3] = rotation

    return Twc


if __name__ == "__main__":
    room_base_json = read_json("data/real_room_0/scene/full/data.json")

    random_num = 100

    # two case: box on the desk, box on the ground ?
    out_dir = "debug/real_room_rand_arrangement/"

    for i in tqdm(range(random_num)):
        finish_arrangement = False
        while not finish_arrangement:
            arrangement_dict = copy.deepcopy(room_base_json)
            arrangement_dict.pop("fix_rot_X", None)
            arrangement_dict.pop("fix_rot_Y", None)
            # random camera
            Twc = random_camera_in_room(
                interp_x=(random.random() * 0.6 + 0.2),
                interp_y=(random.random() * 0.6 + 0.2),
                yaw_angle=(random.random() * 360),
            )
            arrangement_dict["camera"] = {}
            camera_dict = arrangement_dict["camera"]
            camera_dict["cam3d2world"] = Twc.tolist()
            camera_dict["world2cam3d"] = np.linalg.inv(Twc).tolist()
            camera_dict["pos"] = Twc[:3, 3].tolist()
            target = np.array([0, 0, 1])
            target = Twc[:3, 3] + Twc[:3, :3] @ target
            camera_dict["target"] = target.tolist()
            camera_dict["up"] = [0, 0, 1]
            camera_dict["height"] = 512
            camera_dict["width"] = 1024

            arrangement_dict["objs"] = []  # clear objects
            finish_arrangement = True
            for idx, obj_item in enumerate(room_base_json["objs"]):
                obj_id = obj_item["id"]
                if obj_id == 0:
                    continue
                success = False
                not_attach_wall = random.randint(0, 1) == 0
                for i_try in range(50):
                    interp = random.random()
                    layout_idx = random.randint(0, 3)
                    if obj_id == 34 and not_attach_wall:  # not attach wall for desk
                        # if True:
                        interp_y = random.random()
                        if try_place_object_center(
                            obj_item,
                            arrangement_dict["objs"],
                            layout_idx,
                            interp_x=interp,
                            interp_y=interp_y,
                        ):
                            arrangement_dict["objs"] += [obj_item]
                            success = True
                            break
                    else:
                        if try_place_object_attach_wall(
                            obj_item,
                            arrangement_dict["objs"],
                            layout_idx,
                            interp=interp,
                        ):
                            arrangement_dict["objs"] += [obj_item]
                            success = True
                            break
                if not success:
                    print(f"Place {obj_id} failed, try again")
                    finish_arrangement = False
        # if put box(31) on top of the desk(34)
        put_box_on_top = random.randint(0, 1) == 0
        if put_box_on_top:
            # if True:
            box_obj_item = arrangement_dict["objs"][0]
            desk_obj_item = arrangement_dict["objs"][3]
            box_obj_item["bdb3d"]["basis"] = desk_obj_item["bdb3d"]["basis"]
            box_size = box_obj_item["bdb3d"]["size"]
            desk_size = desk_obj_item["bdb3d"]["size"]
            box_obj_item["bdb3d"]["centroid"] = copy.deepcopy(
                desk_obj_item["bdb3d"]["centroid"]
            )
            box_obj_item["bdb3d"]["centroid"][2] = (
                box_obj_item["bdb3d"]["centroid"][2] + (box_size[2] + desk_size[2]) / 2
            )
            arrangement_dict["objs"][0] = box_obj_item
            # print(box_obj_item)
        output_path = os.path.join(out_dir, f"{i:05d}")
        os.makedirs(output_path, exist_ok=True)
        # padding layout
        ref_dict = read_json(
            "data/Beechwood_1_int_bedroom_0_no_random/scene/full/data.json"
        )
        arrangement_dict["layout"] = ref_dict["layout"]
        # padding object info
        for obj_item in arrangement_dict["objs"]:
            obj_item["label"] = -1
            obj_item["is_fixed"] = False
            obj_item["bdb3d"]["scale"] = [1, 1, 1]
        write_json(arrangement_dict, f"{output_path}/data.json")
