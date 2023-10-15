import numpy as np
import torch
import ipdb

# from optim.room_optimizer import RoomOptimizer
from optim.relation_geo import create_grid_points, ray_hit_test
from typing import List, Optional, Any, Dict, Union
from kornia.utils.grid import create_meshgrid
from utils.util import write_point_cloud

# strong prior to ensure relation prediction
attach_wall_types = [
    "bed",
    # "door",
    "window",
    "shelf",
    "nightstand",
    "bottom_cabinet",
    "picture",
    "desk",
    "chair",
    "bench",
]
attach_floor_types = [
    "bed",
    "door",
    "bottom_cabinet",
    "bottom_cabinet_no_top",
    "floor_lamp",
    "chair",
    "office_chair",
    "sofa_chair",
    "table",
    "desk",
    # "shelf",
    "nightstand",
    "stool",
    "carpet",
    "piano",
    "bench",
]
not_attach_floor_type = [
    "window",
    "picture",
    "monitor",
    "table_lamp",
]
not_attach_wall_type = [
    "door",  # due to optimizatio issue
    # "chair",
    "office_chair",
    "sofa_chair",
]

attach_wall_force_direction = [
    "bed",
    "picture",
    "monitor",
    "piano",
    "desk",
    "nightstand",
    "shelf",
    "chair",
    "desk",
]

not_attach_object_type = attach_floor_types


def generate_object_to_room_relation(
    room_optimizer,
    # room_optimizer: RoomOptimizer,
    obj_info: Dict[str, Any],
    ray_grid_size: int = 10,
    ray_grid_stretch: torch.Tensor = torch.ones((1, 3)) * 0.7,
    detect_start_point_offset: float = 0.5,
    distance_th: float = 0.3,
):
    Rwo = obj_info["Rwo"]
    two = obj_info["two"]
    center = two
    rotation = Rwo
    length = torch.Tensor(obj_info["bbox3d"]["size"]).float().cuda() * 0.5

    obj_id = obj_info["obj_id"]
    obj_type = room_optimizer.get_type_of_instance(obj_id)

    face_directions = [
        torch.Tensor([1, 0, 0]),
        torch.Tensor([-1, 0, 0]),
        torch.Tensor([0, 1, 0]),  # back
        torch.Tensor([0, -1, 0]),
        # torch.Tensor([0, 0, 1]), # up
        torch.Tensor([0, 0, -1]),  # down
    ]

    thin_types = ["door", "window"]
    if obj_type in thin_types:
        # for door object, we slightly stretch the size to ensure successive hit-test
        ray_grid_stretch = torch.Tensor([1.2, 1.2, 1])
        # no need to check other direction
        face_directions = [
            torch.Tensor([0, 1, 0]),  # back
            torch.Tensor([0, -1, 0]),
        ]

    face_to_room_distances = []
    hit_point_normals = []

    for face_direction in face_directions:
        grid = create_grid_points(face_direction, grid_size=ray_grid_size)
        face_direction = face_direction.cuda()

        # rays_o for wall collision detection
        # ray shooting from the inner place of the object
        out_detect_start_offset = detect_start_point_offset
        out_detect_rays_d = face_direction
        # scale to fit real size
        out_detect_rays_o = (
            grid * ray_grid_stretch.view(1, 3).cuda() * length
            - face_direction * out_detect_start_offset
        )

        # transform to world coordinate
        out_detect_rays_o = center + (rotation @ out_detect_rays_o.T).T
        out_detect_rays_d = rotation @ out_detect_rays_d

        out_detect_result = ray_hit_test(
            room_optimizer, 0, out_detect_rays_o, out_detect_rays_d, need_normal=True
        )

        face_to_room_distances.append(out_detect_result["depth_fine"])
        hit_point_normals.append(out_detect_result["normals_0"])

        # dump point for debugging
        # if verbose and obj_id == verbose_obj_id:
        if False:
            # if obj_id == 46:
            face_dir_str = f"{int(face_direction[0])}_{int(face_direction[1])}_{int(face_direction[2])}"
            N_rays = out_detect_rays_o.shape[0]
            write_point_cloud(
                out_detect_rays_o.detach().cpu().numpy(),
                f"debug/out_det{obj_id}_dir_{face_dir_str}.ply",
            )
            write_point_cloud(
                (out_detect_rays_o + out_detect_rays_d).detach().cpu().numpy(),
                f"debug/out_det_offset{obj_id}_dir_{face_dir_str}.ply",
            )
            out_hit_pts = out_detect_rays_o + out_detect_rays_d.expand(
                N_rays, 3
            ) * out_detect_result["depth_fine"].view(-1, 1)
            write_point_cloud(
                (out_hit_pts).detach().cpu().numpy(),
                f"debug/out_hit_pts{obj_id}_dir_{face_dir_str}.ply",
            )

    relations = {}
    assert len(face_directions) == len(face_to_room_distances)
    best_cos_angle = 0
    for idx in range(len(face_directions)):
        face_direction = face_directions[idx]
        # choose the 0.1-th nearest depth
        distances = face_to_room_distances[idx].view(-1)
        # hack to make minus depth invalid,
        # since we assume the pose prediction should not be exceed detect_start_point_offset
        distances[distances < 0] = 1e5
        attach_surface_ratio = 0.1
        pt_idx = torch.argsort(distances)[
            int(attach_surface_ratio * distances.shape[0])
        ]
        depth = distances[pt_idx] - out_detect_start_offset
        normal = hit_point_normals[idx][pt_idx]

        # check if satisfy the cos_angle
        cos_angle = torch.dot(-normal.view(-1), Rwo @ face_direction.cuda())
        if cos_angle < np.cos(np.pi / 2):
            # print("continue for cos", cos_angle, obj_id, face_direction)
            continue
        # print("obj_to_room", obj_id, depth, face_direction, cos_angle)

        if depth < distance_th:
            if face_direction[2] == 0:
                relations["attach_wall"] = True
                # only choose one witht the best cos_angle
                if cos_angle > best_cos_angle:
                    relations["attach_wall_face_dir"] = face_direction
                    relations["attach_wall_cos_angle"] = cos_angle
                    best_cos_angle = cos_angle
            else:
                relations["attach_floor"] = True

    if obj_type in attach_wall_types:
        if "attach_wall" not in relations or obj_type in attach_wall_force_direction:
            relations.update(
                {
                    "attach_wall": True,
                    "attach_wall_face_dir": torch.Tensor([0, 1, 0]),  # back
                }
            )
            print(
                f"--> Force add attach_wall relation to {obj_id} with {obj_type} type."
            )
    if obj_type in attach_floor_types:
        if "attach_floor" not in relations:
            relations.update(
                {
                    "attach_floor": True,
                }
            )
            print(
                f"---> Force add attach_floor relation to {obj_id} with {obj_type} type."
            )
    if obj_type in not_attach_wall_type:
        if "attach_wall" in relations:
            relations.pop("attach_wall", None)
            print(
                f"---> Force remove attach_wall relation in {obj_id} with {obj_type} type."
            )
    if obj_type in not_attach_floor_type:
        if "attach_floor" in relations:
            relations.pop("attach_floor", None)
            print(
                f"---> Force remove attach_floor relation in {obj_id} with {obj_type} type."
            )

    return relations


def generate_relation_for_all(
    # room_optimizer: RoomOptimizer,
    room_optimizer,
    all_obj_info: Dict[str, Any],
    obj_to_room_distance_th: float = 0.3,
    top_down_dist_th: float = 0.3,
    top_down_xy_close_factor: float = 0.8,
):
    """
    Inputs:
        top_down_dist_th: when top-down distance lower than this threshould, ...
        top_down_xy_close_factor: top-down close object should also be close in xy-plane, by
            xy_distance < tow_bbox_max_distance * top_down_xy_close_factor
    """
    # ray hit world test, to check whether it attach to floor or wall
    # if close to wall, check type whether it should be close or not
    # e.g. for bed and some furnitures, close -> attach_wall
    # for lamp, close may not be strong enough

    for obj_id_str, obj_info in all_obj_info.items():
        # check object type
        obj_to_room_relation = generate_object_to_room_relation(
            room_optimizer=room_optimizer,
            obj_info=obj_info,
            distance_th=obj_to_room_distance_th,
        )
        room_optimizer.relation_info[obj_id_str] = {}
        room_optimizer.relation_info[obj_id_str].update(obj_to_room_relation)

    # object-object close test
    # check object to object direction and facial alignment
    # e.g. nightstand should be close to some object
    # vase or television should be on the top of a desk ?

    # check upside-down cases
    for src_obj_id_str, src_obj_info in all_obj_info.items():
        # compute source bottom z
        src_two = src_obj_info["two"]
        src_length = torch.Tensor(src_obj_info["bbox3d"]["size"]).float() * 0.5
        src_bottom_z = float(src_two[2]) - float(src_length[2])

        src_type = room_optimizer.get_type_of_instance(int(src_obj_id_str))
        if src_type in not_attach_object_type:
            continue
        for tgt_obj_id_str, tgt_obj_info in all_obj_info.items():
            if src_obj_id_str == tgt_obj_id_str:
                continue
            # compute target top z
            tgt_two = tgt_obj_info["two"]
            tgt_length = torch.Tensor(tgt_obj_info["bbox3d"]["size"]).float() * 0.5
            tgt_top_z = float(tgt_two[2]) + float(tgt_length[2])
            is_topdown_close = abs(src_bottom_z - tgt_top_z) < top_down_dist_th
            # xy distance
            xy_bbox_dist_th = src_length[:2].norm() + tgt_length[:2].norm()
            xy_real_dist = (src_two[:2] - tgt_two[:2]).norm()
            is_xy_dist_small = xy_real_dist < xy_bbox_dist_th * top_down_xy_close_factor

            if is_topdown_close and is_xy_dist_small:
                room_optimizer.relation_info[src_obj_id_str].update(
                    {
                        "attach_bottom_to_object": True,
                        "attach_tgt_obj_id": int(tgt_obj_id_str),
                    }
                )

    # print(room_optimizer.relation_info)
