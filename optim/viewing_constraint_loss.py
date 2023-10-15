import numpy as np
import torch
import ipdb

from typing import List, Optional, Any, Dict, Union
from utils.util import write_point_cloud

# from optim.room_optimizer import RoomOptimizer

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

no_viewing_constraint_types = ["bed"]


def compute_bbox_corner_in_world(obj_info, Rwo, two, reduce_to_one_center=False):
    if reduce_to_one_center:
        bbox_pts = torch.zeros(1, 3).cuda()
    else:
        length = torch.Tensor(obj_info["bbox3d"]["size"]).float().cuda() * 0.5
        bbox_pts = torch.from_numpy(BBOX_PTS).float().cuda() * length.view(1, 3)
    bbox_pts = (Rwo @ bbox_pts.T).T + two
    return bbox_pts


def viewing_constraint_loss(
    # room_optimizer: RoomOptimizer,
    room_optimizer,
    Twc: torch.Tensor,  # camera pose
    all_obj_info: Dict[str, Any],
    # only_center_point=False,
):
    only_center_point = True
    # only_center_point = False
    loss_dict = {}
    for obj_id_str, obj_info in all_obj_info.items():
        if obj_id_str == "0":
            continue
        obj_id = obj_info["obj_id"]
        obj_type = room_optimizer.get_type_of_instance(obj_id)
        if obj_type in no_viewing_constraint_types:
            continue
        # optmize pose
        Rwo = obj_info["Rwo"]
        two = obj_info["two"]
        # initial prediction
        Two_init_pred = room_optimizer.initial_pose_prediction[obj_id_str]["Two"]
        Rwo_init_pred = Two_init_pred[:3, :3]
        two_init_pred = Two_init_pred[:3, 3]
        # TODO: weighted by cos angle?
        bbox_pts_curr = compute_bbox_corner_in_world(
            obj_info, Rwo, two, only_center_point
        )
        bbox_pts_init_pred = compute_bbox_corner_in_world(
            obj_info, Rwo_init_pred, two_init_pred, only_center_point
        )
        viewing_dir_curr = bbox_pts_curr - Twc[:3, 3]
        viewing_dir_init_pred = bbox_pts_init_pred - Twc[:3, 3]
        cos = torch.nn.CosineSimilarity(dim=1)
        loss_dict[f"viewing_constraint_loss_{obj_id}"] = (
            1 - cos(viewing_dir_curr, viewing_dir_init_pred)
        ).sum() * 1e2
        # ).sum()

        x_vec = torch.Tensor([1, 0, 0]).float().cuda()
        # ((Rwo_init_pred @ Rwo) @ torch.Tensor([0, 1, 0]).cuda() ).dot( torch.Tensor([0, 1, 0])
        loss_x_axis = (((Rwo_init_pred.T @ Rwo) @ x_vec).dot(x_vec) - 1).abs()
        loss_dict[f"viewing_rotation_loss_{obj_id}"] = loss_x_axis * 1e1
    # print(loss_dict)
    return loss_dict
