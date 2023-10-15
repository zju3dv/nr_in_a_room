import numpy as np
import torch

from typing import List, Optional, Any, Dict, Union
from kornia.utils.grid import create_meshgrid
from utils.util import write_point_cloud
from models_neurecon.neus_multi_rendering import sphere_tracing_surface_points

# from optim.room_optimizer import RoomOptimizer


def get_random_points_on_sphere(N, device, eps=1e-5):
    rand_sphere_pts = torch.randn((N, 3), device=device)
    rand_sphere_pts = torch.nn.functional.normalize(rand_sphere_pts, dim=1, eps=eps)
    return rand_sphere_pts


def sample_inner_points(
    room_optimizer,
    obj_info: Dict[str, Any],
    N_sample_points: int,
    sample_at_surface: bool = True,
    # surface_sample_inside_th: float = -0.01,  # sphere tracing until hit point near the
    N_sphere_tracing_iter: int = 20,
    cache_sampling: bool = True,
):
    obj_id = obj_info["obj_id"]
    with torch.no_grad():
        # cache sampling
        if not hasattr(sample_inner_points, "cache_sample"):
            sample_inner_points.cache_sample = {}
        if (
            cache_sampling and str(obj_id) in sample_inner_points.cache_sample
        ):  # if object centric local points has been sampled, we directly use cached results
            rand_pts = sample_inner_points.cache_sample[str(obj_id)].cuda()
        else:  # we need to sample inside the object
            # bbox length
            length = torch.Tensor(obj_info["bbox3d"]["size"]).float().cuda()
            # condition code
            obj_code = room_optimizer.image_attrs[str(obj_id)].embedding_instance(
                torch.ones((1)).long().cuda() * obj_id
            )
            rand_pts = torch.empty((0, 3), device="cuda")
            if sample_at_surface:  # sample on the surface via
                # print("Sampling inner points for", obj_id)
                while rand_pts.shape[0] < N_sample_points:
                    # print("sampling for ", obj_id)
                    # randomly sample on a uniform sphere
                    new_rand_pts = get_random_points_on_sphere(
                        int(N_sample_points * 2), device="cuda"
                    )
                    bbox_diagonal_dist = (
                        float(length.norm())
                        * 1.2
                        / room_optimizer.get_scale_factor(obj_id)
                    )
                    rays_o = new_rand_pts * (
                        bbox_diagonal_dist / 2
                    )  # stretch following the bbox size
                    # write_point_cloud(
                    #     rays_o.detach().cpu().numpy(), "debug/obj_local_rays_o.ply"
                    # )
                    # generate rays towards object center with slightly direction perturb (np.pi / 6)
                    rays_d_to_center = -new_rand_pts
                    rays_d = get_random_points_on_sphere(rays_o.shape[0], device="cuda")
                    mask = torch.zeros_like(rays_d[:, 0]).bool()
                    ray_th = np.cos(np.pi / 6)
                    # TODO(ybbbbt): this may be slow
                    while True:
                        mask = torch.mul(rays_d, rays_d_to_center).sum(-1) > ray_th
                        if (~mask).sum() == 0:
                            break
                        rays_d[~mask] = get_random_points_on_sphere(
                            (~mask).sum(), device="cuda"
                        )

                    _, new_rand_pts, mask, sdf = sphere_tracing_surface_points(
                        room_optimizer.models[f"neus_{obj_id}"].implicit_surface,
                        rays_o=rays_o,
                        rays_d=rays_d,
                        obj_code=obj_code.expand(rays_o.shape[0], -1),
                        near=torch.zeros_like(rays_o[:, 0]),
                        far=torch.ones_like(rays_o[:, 0]) * bbox_diagonal_dist,
                        # stop_sdf_th=surface_sample_inside_th,
                        N_iters=N_sphere_tracing_iter,
                    )
                    # remove points far from the surface, since some points may fails to hit
                    mask = torch.logical_and(mask, sdf.abs() < 5e-3)
                    new_rand_pts = new_rand_pts[mask]
                    # remove points out of bbox bound
                    # TODO(ybbbbt): we assume bbox at the object center
                    # bbox_margin_eps = 0.025
                    # bbox_max, bbox_min = (
                    #     length / 2 + bbox_margin_eps,
                    #     -length / 2 - bbox_margin_eps,
                    # )
                    # mask = torch.logical_or(
                    #     new_rand_pts > bbox_max.view(1, 3),
                    #     new_rand_pts < bbox_min.view(1, 3),
                    # )
                    # mask = mask.float().sum(axis=1) == 0
                    # new_rand_pts = new_rand_pts[mask]

                    rand_pts = torch.cat([rand_pts, new_rand_pts], dim=0)
                sample_inner_points.cache_sample[str(obj_id)] = rand_pts.detach().cpu()
            else:  # sample randomly inside the bounding box
                while rand_pts.shape[0] < N_sample_points:
                    new_rand_pts = torch.rand(
                        (int(N_sample_points * 1.5), 3), device="cuda"
                    )
                    # fit to bbox
                    new_rand_pts = (new_rand_pts - 0.5) * length
                    # inference sdf
                    sdf = room_optimizer.models[
                        f"neus_{obj_id}"
                    ].implicit_surface.forward(
                        new_rand_pts, obj_code.expand(new_rand_pts.shape[0], -1)
                    )
                    mask_inside = sdf < 0
                    new_rand_pts = new_rand_pts[mask_inside]
                    rand_pts = torch.cat([rand_pts, new_rand_pts], dim=0)
                rand_pts = rand_pts[:N_sample_points]
                sample_inner_points.cache_sample[str(obj_id)] = rand_pts.detach().cpu()
            # write_point_cloud(
            #     rand_pts.detach().cpu().numpy(), f"debug/obj_local_{obj_id}.ply"
            # )

    # convert to world coordinate
    Rwo = obj_info["Rwo"]
    two = obj_info["two"]
    rand_pts = (Rwo @ (rand_pts * room_optimizer.get_scale_factor(obj_id)).T).T + two
    # write_point_cloud(rand_pts.detach().cpu().numpy(), "debug/obj_global.ply")
    # exit(0)

    return rand_pts


def compute_sdf_in_object(
    room_optimizer, obj_info: Dict[str, Any], sample_pts: torch.Tensor
):
    obj_id = obj_info["obj_id"]
    if obj_id == 0:  # background
        sample_pts = (
            sample_pts - torch.from_numpy(room_optimizer.bg_scene_center).float().cuda()
        ) / room_optimizer.bg_scale_factor
        # write_point_cloud(sample_pts.detach().cpu().numpy(), "debug/test_bg_local.ply")
    else:  # other objects
        # convert to object local coordinate
        Rwo = obj_info["Rwo"]
        two = obj_info["two"]
        sample_pts = (
            Rwo.T @ ((sample_pts / room_optimizer.get_scale_factor(obj_id)) - two).T
        ).T

    obj_code = room_optimizer.image_attrs[str(obj_id)].embedding_instance(
        torch.ones((1)).long().cuda() * obj_id
    )
    sdf = room_optimizer.models[f"neus_{obj_id}"].implicit_surface.forward(
        sample_pts, obj_code.expand(sample_pts.shape[0], -1)
    )
    return sdf


def object_max_dist(obj_info_src, obj_info_dst):
    length_src = np.array(obj_info_src["bbox3d"]["size"])
    length_dst = np.array(obj_info_dst["bbox3d"]["size"])
    return np.linalg.norm(length_src) + np.linalg.norm(length_dst)


def physical_violation_loss(
    room_optimizer,
    all_obj_info: Dict[str, Any],
    N_nearest_obj: int = 3,
    check_background_violation: bool = True,
    N_sample_points: int = 1000,
):
    """
    Inputs:
        check_background_violation: also check whether object_room violation
    """
    # vio_loss = 0
    vio_loss = {}
    if check_background_violation:  # set background as object 0
        all_obj_info["0"] = {"obj_id": 0}
    # get all object ids
    obj_ids = all_obj_info.keys()
    # go through object
    for src_obj_id in obj_ids:
        if src_obj_id == "0":
            continue
        # find N nearest object
        candidate_obj_ids = []
        candidate_dists = []
        # compute distance
        for tgt_obj_id in obj_ids:
            if src_obj_id == tgt_obj_id or tgt_obj_id == "0":
                continue
            candidate_obj_ids += [tgt_obj_id]
            candidate_dists += [
                (
                    all_obj_info[src_obj_id]["two"] - all_obj_info[tgt_obj_id]["two"]
                ).norm()
            ]
        nearest_obj_ids = []
        # order by distance
        N_candidates = len(candidate_obj_ids)
        if N_candidates > 0:
            candidate_dists = torch.stack(candidate_dists, 0)
            nearest_ind = torch.argsort(candidate_dists)[
                : min(N_nearest_obj, N_candidates)
            ]
            for idx, obj_id in enumerate(candidate_obj_ids):
                if (
                    object_max_dist(all_obj_info[src_obj_id], all_obj_info[obj_id])
                    > candidate_dists[idx]
                ):  # skip with too long distance
                    continue
                if idx in nearest_ind:
                    nearest_obj_ids += [obj_id]
        # add background to violation loss
        if check_background_violation:
            nearest_obj_ids += ["0"]

        # sample inner points for current active object
        sample_pts = sample_inner_points(
            room_optimizer=room_optimizer,
            obj_info=all_obj_info[src_obj_id],
            N_sample_points=N_sample_points,
        )

        # convert to target object space and test
        for test_obj_id in nearest_obj_ids:
            sdf = compute_sdf_in_object(
                room_optimizer, all_obj_info[test_obj_id], sample_pts
            )
            # if test_obj_id == 0:
            #     curr_vio_loss = (
            #         torch.relu(-sdf).mean() * 0.1
            #     )  # too strong violation to the room may introduce side effect?
            # else:
            sdf_eps = 0.025
            curr_vio_loss = torch.relu(-sdf - sdf_eps).sum()
            # if src_obj_id == "13":
            # print(
            #     f"src_obj: {src_obj_id} -> test_obj: {test_obj_id} => {float(curr_vio_loss)}"
            # )
            # vio_loss = vio_loss + curr_vio_loss
            vio_loss[f"phy_vio_loss_{src_obj_id}_{test_obj_id}"] = curr_vio_loss

    return vio_loss
