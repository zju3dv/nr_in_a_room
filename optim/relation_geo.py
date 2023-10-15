import ipdb
import numpy as np
import torch

# from optim.room_optimizer import RoomOptimizer
from typing import List, Optional, Any, Dict, Union
from kornia.utils.grid import create_meshgrid
from utils.util import write_point_cloud


def face_dir_to_str(face_dir: torch.Tensor):
    return f"{float(face_dir[0]):.1f}_{float(face_dir[1]):.1f}_{float(face_dir[2]):.1f}"


def ray_hit_test(
    room_optimizer,
    obj_id: int,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    need_normal=False,
):
    """
    Hit objects with given rays
    Input rays shold be in object coordinate with real scale (unscaled)
    """
    N_rays = rays_o.shape[0]
    if rays_d.numel() == 3:
        rays_d = rays_d.expand((N_rays, 3))

    # generate rays
    # if background (wall) rays, we shift the scene center and rescale the rays
    if obj_id == 0:
        rays_o = (
            rays_o - torch.from_numpy(room_optimizer.bg_scene_center).float().cuda()
        )

    # we use a smaller near far for ray hit test
    rays = room_optimizer.generate_object_rays(rays_o, rays_d, obj_id, near=0, far=3)

    # prepare object embedding code
    extra_dict = {}
    extra_dict["embedding_inst_{}".format(obj_id)] = room_optimizer.image_attrs[
        str(obj_id)
    ].embedding_instance(torch.ones_like(rays_o[..., 0]).long().cuda() * obj_id)

    # render rays
    result = room_optimizer.batched_inference_multi(
        [rays],
        [obj_id],
        to_cpu=False,
        hit_test_only=True,
        need_normal=need_normal,
        **extra_dict,
    )
    # unpack output to hit probability, depth and point3d
    return result


def create_grid_points(
    face_direction: torch.Tensor,
    grid_size: int = 15,
):
    """
    create grid points as shooting ray array
    """
    # create mesh grid for box sampling
    grid = create_meshgrid(
        grid_size, grid_size, normalized_coordinates=True, device="cuda"
    )  # [1, grid_size, grid_size, 2]
    grid = grid.squeeze(0)  # [grid_size, grid_size, 2]

    for i_axis in range(3):
        if face_direction[i_axis] != 0:
            grid = grid.unsqueeze(i_axis)
            ones = torch.ones_like(grid[..., 0:1]) * face_direction[i_axis]
            first_half = grid[..., 0:i_axis]
            second_half = grid[..., i_axis:2]
            grid = torch.cat([first_half, ones, second_half], dim=-1)
            break
    # dirty fix grid shape
    if len(grid.shape) == 4:
        grid = grid.squeeze(0)
    # h, w, d, _ = grid.shape
    if face_direction[2] == 0:
        # when not in top down direction, we crop the sampling near the floor
        grid = grid[int(grid_size * 0.3) :, :, :]
    # rescale grid to fit bbox
    grid = grid.reshape(-1, 3)
    return grid


def raycast_object_attach_room(
    room_optimizer,
    obj_info: Dict[str, Any],
    face_direction: torch.Tensor,
    ray_grid_size: int,
    ray_grid_stretch: torch.Tensor = torch.ones((1, 3)),
    ray_grid_offset: torch.Tensor = None,
    use_bbox_surface_as_in_detect: bool = False,
    detect_start_point_offset: float = 0.5,
    cache_obj_in_hit_result=True,
    verbose=False,
    verbose_obj_id=5,
):
    """
    we assume that face_direction is normalized as [1, 0, 0]
    Inputs:
        ray_grid_stretch: stretch so as to ensure successive hit-test for door (where the attach wall may be empty)
    """
    """ generate hit rays according to face_direction"""
    Rwo = obj_info["Rwo"]
    two = obj_info["two"]
    center = two
    rotation = Rwo
    length = torch.Tensor(obj_info["bbox3d"]["size"]).float().cuda() * 0.5

    obj_id = obj_info["obj_id"]

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

    if isinstance(ray_grid_offset, torch.Tensor):
        out_detect_rays_o += ray_grid_offset.view(1, 3).cuda()

    # transform to world coordinate
    out_detect_rays_o = center + (rotation @ out_detect_rays_o.T).T
    out_detect_rays_d = rotation @ out_detect_rays_d

    out_detect_result = ray_hit_test(
        room_optimizer, 0, out_detect_rays_o, out_detect_rays_d
    )

    # dump point for debugging
    if verbose and obj_id == verbose_obj_id:
        N_rays = out_detect_rays_o.shape[0]
        write_point_cloud(
            out_detect_rays_o.detach().cpu().numpy(), f"debug/out_det{obj_id}.ply"
        )
        write_point_cloud(
            (out_detect_rays_o + out_detect_rays_d).detach().cpu().numpy(),
            f"debug/out_det_offset{obj_id}.ply",
        )
        out_hit_pts = out_detect_rays_o + out_detect_rays_d.expand(
            N_rays, 3
        ) * out_detect_result["depth_fine"].view(-1, 1)
        write_point_cloud(
            (out_hit_pts).detach().cpu().numpy(), f"debug/out_hit_pts{obj_id}.ply"
        )

    in_detect_start_offset = (
        0 if use_bbox_surface_as_in_detect else detect_start_point_offset
    )
    # cache to speed up
    cache_key = f"{obj_id}_{face_dir_to_str(face_direction)}"
    if not hasattr(raycast_object_attach_room, "cache_in_detect_res"):
        raycast_object_attach_room.cache_in_detect_res = {}
    if (
        not cache_obj_in_hit_result
        or cache_key not in raycast_object_attach_room.cache_in_detect_res
    ):
        with torch.no_grad():
            if use_bbox_surface_as_in_detect:
                N_rays = grid.shape[0]
                in_detect_result = {
                    "depth_fine": torch.zeros((N_rays, 1)).cuda(),
                    "opacity_fine": torch.ones((N_rays, 1)).cuda(),
                }
            else:
                # ray shooting from the outer place of the object
                in_detect_rays_d = -face_direction
                # apply offset, so that it fit object rendering offset inside the object-nerf model
                in_detect_rays_o = (
                    grid * length + face_direction * in_detect_start_offset
                )
                in_detect_result = ray_hit_test(
                    room_optimizer, obj_id, in_detect_rays_o, in_detect_rays_d
                )
            raycast_object_attach_room.cache_in_detect_res[cache_key] = in_detect_result
    else:
        in_detect_result = raycast_object_attach_room.cache_in_detect_res[cache_key]

    # dump in_detect results
    # print(obj_id)
    if verbose and obj_id == verbose_obj_id:
        # if obj_id == 4:
        # ipdb.set_trace()
        in_detect_rays_o = in_detect_rays_o
        in_detect_rays_o = center + (rotation @ in_detect_rays_o.T).T
        in_detect_rays_d = rotation @ in_detect_rays_d
        ipdb.set_trace()
        """dump point for debugging"""
        write_point_cloud(
            in_detect_rays_o.detach().cpu().numpy(), f"debug/in_det{obj_id}.ply"
        )
        write_point_cloud(
            (in_detect_rays_o + in_detect_rays_d).detach().cpu().numpy(),
            f"debug/in_det_offset{obj_id}.ply",
        )
        mask = (in_detect_result["opacity_fine"] > 0.8).squeeze()
        depth = in_detect_result["depth_fine"][mask]
        print(depth.shape)
        in_detect_rays_o = in_detect_rays_o[mask]
        N_rays = in_detect_rays_o.shape[0]

        in_hit_pts = in_detect_rays_o + in_detect_rays_d.expand(
            N_rays, 3
        ) * depth.reshape(-1, 1)
        write_point_cloud(
            (in_hit_pts).detach().cpu().numpy(), f"debug/in_hit_pts{obj_id}.ply"
        )

    return {
        "in_det_res": in_detect_result,
        "out_det_res": out_detect_result,
        # base distance of in-out shooting rays
        "base_dist_in": in_detect_start_offset,
        "base_dist_out": out_detect_start_offset,
    }


#
increase_detect_offset = {}


def aggressive_detect_offset(unique_key, max_offset=1.5, increase_step=0.1):
    if unique_key not in increase_detect_offset:
        increase_detect_offset[unique_key] = increase_step
    elif increase_detect_offset[unique_key] < max_offset:
        increase_detect_offset[unique_key] += increase_step
    print(
        f"Increase {unique_key} detect offset to",
        increase_detect_offset[unique_key],
    )


def object_room_magnetic_loss(
    room_optimizer,
    obj_info: Dict[str, Any],
    face_direction: torch.Tensor,
    ray_grid_stretch: torch.Tensor = torch.ones((1, 3)),
    ray_grid_offset: torch.Tensor = None,
    use_bbox_surface_as_in_detect: bool = False,
    ray_grid_size: int = 15,
):
    """
    use_bbox_surface_as_in_detect:
        Do not compute surface by raycasting. Instead, we directly use bbox surface as grid array.
        This is useful for the loss like "bottom attach to floor",
        since bottom may be unobservable and thus not reconstructed properly.

    """
    obj_id = obj_info["obj_id"]
    unique_key = f"{obj_id}_{face_dir_to_str(face_direction)}"
    raycast_res = raycast_object_attach_room(
        room_optimizer=room_optimizer,
        obj_info=obj_info,
        face_direction=face_direction,
        ray_grid_size=ray_grid_size,
        ray_grid_stretch=ray_grid_stretch,
        ray_grid_offset=ray_grid_offset,
        use_bbox_surface_as_in_detect=use_bbox_surface_as_in_detect,
        detect_start_point_offset=0.5 + increase_detect_offset.get(unique_key, 0),
    )

    in_det_res = raycast_res["in_det_res"]
    out_det_res = raycast_res["out_det_res"]

    base_dist_in = raycast_res["base_dist_in"]
    base_dist_out = raycast_res["base_dist_out"]
    in_detect_hit_prob = in_det_res["opacity_fine"]
    in_detect_depth = in_det_res["depth_fine"]
    out_detect_hit_prob = out_det_res["opacity_fine"]
    out_detect_depth = out_det_res["depth_fine"]

    # for the empty part, we do not consider it
    out_detect_mask = out_detect_hit_prob > 0.8
    in_detect_mask = in_detect_hit_prob > 0.8
    if not use_bbox_surface_as_in_detect:
        mask = torch.logical_and(in_detect_mask, out_detect_mask)
    else:
        mask = out_detect_mask

    in_detect_depth = in_detect_depth[mask]
    out_detect_depth = out_detect_depth[mask]

    if use_bbox_surface_as_in_detect:
        thin_types = ["door", "window"]
        if obj_info["obj_type"] in thin_types:  # hack for door and window type
            # choose from the outer shoot (e.g. edge of door)
            attach_surface_ratio = 0.05
            if out_detect_depth.shape[0] == 0:
                ipdb.set_trace()
            depth_th = torch.sort(out_detect_depth)[0][
                int(attach_surface_ratio * out_detect_depth.shape[0])
            ]
            # sometimes, the object may be slightly tilted, so we loose the depth_th
            depth_th += 0.05
            # print(depth_th)
            mask_contact = out_detect_depth < depth_th
            # also compensate thickness of the door, to make it aligned better
            length = torch.Tensor(obj_info["bbox3d"]["size"]).float()
            if face_direction[2] == 0:
                base_dist_in = -torch.matmul(face_direction, length).abs().sum()
            # print("base_dist_in", base_dist_in)
        else:
            # we assume all the points are valid attachment
            mask_contact = torch.ones_like(in_detect_depth).bool()
            if in_detect_depth.numel() == 0:
                print(
                    "Warning: attach failed for key {}, in_detect_mask.sum() = {}, out_detect_mask.sum() = {}".format(
                        unique_key, in_detect_mask.sum(), out_detect_mask.sum()
                    )
                )
                aggressive_detect_offset(unique_key)
                return {}
    else:
        if in_detect_depth.numel() == 0:
            # ipdb.set_trace()
            # raise RuntimeError(
            #     f"Not enough points for obj={obj_id}, maybe checkpoint or obj_id is wrong."
            # )
            print(
                "Warning: attach failed for key {}, in_detect_mask.sum() = {}, out_detect_mask.sum() = {}".format(
                    unique_key, in_detect_mask.sum(), out_detect_mask.sum()
                )
            )
            aggressive_detect_offset(unique_key)
            return {}
        # else:
        #     # we can safely reduce detect offset when it find attachment
        #     if (
        #         unique_key in increase_detect_offset
        #         and increase_detect_offset[unique_key] > 0
        #     ):
        #         increase_detect_offset[unique_key] -= 0.025

        # only choose the nearest depth as attach surface points
        attach_surface_ratio = 0.1
        depth_th = torch.sort(in_detect_depth)[0][
            int(attach_surface_ratio * in_detect_depth.shape[0])
        ]
        # sometimes, the object may be slightly tilted, so we loose the depth_th
        depth_th += 0.05
        mask_contact = in_detect_depth < depth_th

    in_detect_depth_attach = in_detect_depth[mask_contact]
    out_detect_depth_attach = out_detect_depth[mask_contact]

    attach_loss = torch.relu(
        out_detect_depth_attach + in_detect_depth_attach - base_dist_in - base_dist_out
    ).mean()

    if room_optimizer.get_type_of_instance(obj_id) in ["picture", "mirror"]:
        violation_loss = torch.relu(
            base_dist_in + base_dist_out - out_detect_depth - in_detect_depth
        ).sum()
    else:
        violation_loss = torch.relu(
            base_dist_in + base_dist_out - out_detect_depth - in_detect_depth
        ).mean()

    # magnetic_loss = (
    #     (
    #         out_detect_depth_attach
    #         + in_detect_depth_attach
    #         - base_dist_in
    #         - base_dist_out
    #     )
    #     .pow(2)
    #     .sum()
    # )

    # if obj_id == 62:
    #     print("attach", attach_loss, "violation", violation_loss)
    # import ipdb

    # ipdb.set_trace()
    # print(magnetic_loss)

    # TODO(ybbbbt): make weights configurable
    # print(unique_key, attach_loss, violation_loss)
    return {
        f"raycast_attach_loss_{unique_key}": attach_loss,
        f"raycast_violation_loss_{unique_key}": violation_loss,
        # f"magnetic_loss_{unique_key}": magnetic_loss
        # * 10,
    }


def raycast_object_close_to_object(
    room_optimizer,
    obj_info_src: Dict[str, Any],
    obj_info_tgt: Dict[str, Any],
    face_direction: torch.Tensor,
    ray_grid_size: int = 15,
    cache_obj_in_hit_result=True,
    verbose=False,
    verbose_obj_id=5,
):
    """
    shoot rays from obj_src to obj_tgt
    face_direction: in which face source object is close to target object
        we assume that face_direction is normalized as [1, 0, 0]
    """
    """ generate hit rays according to face_direction"""
    Rwo_src = obj_info_src["Rwo"]
    two_src = obj_info_src["two"]
    length_src = torch.Tensor(obj_info_src["bbox3d"]["size"]).float().cuda() * 0.5
    Rwo_tgt = obj_info_tgt["Rwo"]
    two_tgt = obj_info_tgt["two"]
    # T_o_tgt_o_src
    R_o_tgt_src = Rwo_tgt.T @ Rwo_src
    t_o_tgt_src = Rwo_tgt.T @ (two_src - two_tgt)

    obj_id_src = obj_info_src["obj_id"]
    obj_id_tgt = obj_info_tgt["obj_id"]

    # create mesh grid for box sampling
    grid = create_grid_points(face_direction, grid_size=ray_grid_size)
    face_direction = face_direction.cuda()

    """Ray shoot to source object"""
    in_detect_start_offset = 0.5
    # cache to speed up
    cache_key = f"{obj_id_src}_{face_dir_to_str(face_direction)}"
    if not hasattr(raycast_object_close_to_object, "cache_in_detect_res"):
        raycast_object_close_to_object.cache_in_detect_res = {}
    if (
        not cache_obj_in_hit_result
        or cache_key not in raycast_object_close_to_object.cache_in_detect_res
    ):
        with torch.no_grad():
            # ray shooting from the outer place of the object
            in_detect_rays_d = -face_direction
            # apply offset, so that it fit object rendering offset inside the object-nerf model
            in_detect_rays_o = (
                grid * length_src + face_direction * in_detect_start_offset
            )
            in_detect_result = ray_hit_test(
                room_optimizer, obj_id_src, in_detect_rays_o, in_detect_rays_d
            )
            raycast_object_close_to_object.cache_in_detect_res[
                cache_key
            ] = in_detect_result
    else:
        in_detect_result = raycast_object_close_to_object.cache_in_detect_res[cache_key]

    """Ray shoot to target object"""
    # rays_o for wall collision detection
    # ray shooting from the inner place of the object
    out_detect_start_offset = 0.5
    out_detect_rays_d = face_direction
    # scale to fit real size
    # use source shape to generate out detect rays_o
    out_detect_rays_o = grid * length_src - face_direction * out_detect_start_offset
    # transform from source object space to target object space
    out_detect_rays_o = t_o_tgt_src + (R_o_tgt_src @ out_detect_rays_o.T).T
    out_detect_rays_d = R_o_tgt_src @ out_detect_rays_d

    out_detect_result = ray_hit_test(
        room_optimizer, obj_id_tgt, out_detect_rays_o, out_detect_rays_d
    )

    return {
        "in_det_res": in_detect_result,
        "out_det_res": out_detect_result,
        # base distance of in-out shooting rays
        "base_dist_in": in_detect_start_offset,
        "base_dist_out": out_detect_start_offset,
    }


def object_object_attach_loss(
    room_optimizer,
    obj_info_src: Dict[str, Any],
    obj_info_tgt: Dict[str, Any],
    face_direction: torch.Tensor,
):
    obj_id_src = obj_info_src["obj_id"]
    obj_id_tgt = obj_info_tgt["obj_id"]
    raycast_res = raycast_object_close_to_object(
        room_optimizer, obj_info_src, obj_info_tgt, face_direction
    )
    in_det_res = raycast_res["in_det_res"]
    out_det_res = raycast_res["out_det_res"]

    base_dist_in = raycast_res["base_dist_in"]
    base_dist_out = raycast_res["base_dist_out"]
    in_detect_hit_prob = in_det_res["opacity_fine"]
    in_detect_depth = in_det_res["depth_fine"]
    out_detect_hit_prob = out_det_res["opacity_fine"]
    out_detect_depth = out_det_res["depth_fine"]

    # for the empty part, we do not consider it
    in_detect_mask = in_detect_hit_prob > 0.8
    out_detect_mask = out_detect_hit_prob > 0.8
    mask = torch.logical_and(in_detect_mask, out_detect_mask)

    in_detect_depth = in_detect_depth[mask]
    out_detect_depth = out_detect_depth[mask]

    if in_detect_depth.numel() == 0:
        print(f"Object-Object: Not enough points for obj_src={obj_id_src}.")
        return {}
    if out_detect_depth.numel() == 0:
        print(f"Object-Object: Not enough points for obj_tgt={obj_id_tgt}.")
        return {}

    # only choose the nearest depth as attach surface points
    attach_surface_ratio = 0.1
    depth_th_in = torch.sort(in_detect_depth)[0][
        int(attach_surface_ratio * in_detect_depth.shape[0])
    ]
    depth_th_out = torch.sort(out_detect_depth)[0][
        int(attach_surface_ratio * out_detect_depth.shape[0])
    ]
    mask_contact = torch.logical_and(
        in_detect_depth < depth_th_in, out_detect_depth < depth_th_out
    )

    in_detect_depth_attach = in_detect_depth[mask_contact]
    out_detect_depth_attach = out_detect_depth[mask_contact]

    attach_loss = torch.relu(
        out_detect_depth_attach + in_detect_depth_attach - base_dist_in - base_dist_out
    ).sum()

    violation_loss = torch.relu(
        base_dist_in + base_dist_out - out_detect_depth - in_detect_depth
    ).mean()

    # ipdb.set_trace()
    obj_obj_attach_loss = attach_loss + violation_loss

    # print("obj-obj", obj_id_src, obj_id_tgt, float(obj_obj_attach_loss))

    # TODO(ybbbbt): make weights configurable
    return {
        "obj_obj_attach_loss_{}_{}".format(obj_id_src, obj_id_tgt): obj_obj_attach_loss,
    }
