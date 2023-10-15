import ipdb
import torch
import sys
import os

sys.path.append(os.getcwd())  # noqa
import copy
from typing import List, Dict, Any
from einops import rearrange, reduce, repeat
from models_neurecon.neus import NeuS, volume_render
from models_neurecon.base import ImplicitSurface


def volume_rendering_multi_neus(
    results,
    typ,
    z_vals_list,
    rgbs_list,
    alphas_list,
    noise_std,
    white_back,
    obj_ids_list=None,
):
    N_objs = len(z_vals_list)
    # order via z_vals
    z_vals = torch.cat(z_vals_list, 1)  # (N_rays, N_samples*N_objs)
    rgbs = torch.cat(rgbs_list, 1)  # (N_rays, N_samples*N_objs, 3)
    alphas = torch.cat(alphas_list, 1)  # (N_rays, N_samples*N_objs)

    z_vals, idx_sorted = torch.sort(z_vals, -1)
    for i in range(3):
        rgbs[:, :, i] = torch.gather(rgbs[:, :, i].clone(), dim=1, index=idx_sorted)
    alphas = torch.gather(alphas, dim=1, index=idx_sorted)
    # record object ids for recovering weights of each object after sorting
    if obj_ids_list != None:
        obj_ids = torch.cat(obj_ids_list, -1)
        results[f"obj_ids_{typ}"] = torch.gather(obj_ids, dim=1, index=idx_sorted)

    alphas_shifted = torch.cat(
        [torch.ones_like(alphas[:, :1]), 1 - alphas + 1e-10], -1
    )  # [1, 1-a1, 1-a2, ...]
    weights = alphas * torch.cumprod(alphas_shifted[:, :-1], -1)  # (N_rays, N_samples_)

    weights_sum = reduce(
        weights, "n1 n2 -> n1", "sum"
    )  # (N_rays), the accumulated opacity along the rays
    # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically

    # results[f"weights_{typ}"] = weights
    results[f"opacity_{typ}"] = weights_sum
    # results[f"z_vals_{typ}"] = z_vals

    rgb_map = reduce(
        rearrange(weights, "n1 n2 -> n1 n2 1") * rgbs, "n1 n2 c -> n1 c", "sum"
    )
    depth_map = reduce(weights * z_vals, "n1 n2 -> n1", "sum")

    if white_back:
        rgb_map = rgb_map + 1 - weights_sum.unsqueeze(-1)

    results[f"rgb_{typ}"] = rgb_map
    results[f"depth_{typ}"] = depth_map


# adopt from neurecon/ray_casting.py
def sphere_tracing_surface_points(
    implicit_surface: ImplicitSurface,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    # function config
    obj_code: torch.Tensor,
    near: torch.Tensor,
    far: torch.Tensor,
    # algorithm config
    # stop_sdf_th: float = 0.0,
    # N_iters: int = 20,
    N_iters: int = 50,
    near_surface_th: float = 0.0,
    sdf_eps: float = 5e-3,
):
    """
    rays_o, rays_d: torch.Tensor [N_rays, 3]
    obj_code: torch.Tensor [N_rays, N_channel]
    near: torch.Tensor [N_rays]
    far: torch.Tensor [N_rays]
    near_surface_th: also set the output mask to false when hit point not near the surface
    """
    device = rays_o.device
    if isinstance(near, float):
        d_preds = torch.ones([*rays_o.shape[:-1]], device=device) * near
    else:
        d_preds = near
    mask = torch.ones_like(d_preds, dtype=torch.bool, device=device)
    N_rays = d_preds.shape[0]
    for _ in range(N_iters):
        pts = rays_o + rays_d * d_preds[..., :, None]
        surface_val = implicit_surface.forward(pts, obj_code)
        # surface_val = surface_val - stop_sdf_th
        # d_preds[mask] += surface_val[mask]
        d_preds = d_preds + surface_val * mask.float()
        mask[d_preds > far] = False
        mask[d_preds < 0] = False
        # mark unfinished
        mask_unfinish = surface_val.abs() > sdf_eps
        mask_unfinish[~mask] = False
        if mask_unfinish.sum() == 0:
            # print(_)
            break
    pts = rays_o + rays_d * d_preds[..., :, None]
    if near_surface_th != 0:
        mask = torch.logical_and(mask, surface_val.abs() < near_surface_th)
    return d_preds, pts, mask, surface_val


def sphere_tracing_rendering(
    model: NeuS,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    near: torch.Tensor,
    far: torch.Tensor,
    obj_code: torch.Tensor,
    light_code: torch.Tensor,
    appearance_code: torch.Tensor,
    hit_test_only: bool,
    need_normal: bool,
    refine_edge: bool,
    chunk: int,
):
    d_pred_chunk = []
    rgb_chunk = []
    normal_chunk = []
    alpha_chunk = []
    # pt_pred_chunk = []
    # mask_chunk = []
    B = rays_o.shape[0]
    for i in range(0, B, chunk):
        d_pred, pt_pred, mask, last_sdf = sphere_tracing_surface_points(
            implicit_surface=model.implicit_surface,
            rays_o=rays_o[i : i + chunk],
            rays_d=rays_d[i : i + chunk],
            near=near[i : i + chunk].squeeze(1),
            far=far[i : i + chunk].squeeze(1),
            obj_code=obj_code[i : i + chunk],
            near_surface_th=0.05 if hit_test_only else 0,
        )
        d_pred_chunk += [d_pred]
        alpha = torch.zeros_like(d_pred)
        alpha[mask] = 1
        alpha_chunk += [alpha]
        if not hit_test_only:
            rgb, sdf, nablas = model.forward(
                pt_pred,
                obj_code[i : i + chunk],
                None if light_code is None else light_code[i : i + chunk],
                rays_d[i : i + chunk],
                None if appearance_code is None else appearance_code[i : i + chunk],
            )
            rgb_chunk += [rgb]
        if need_normal or refine_edge:
            _, normal, _ = model.implicit_surface.forward_with_nablas(
                pt_pred, obj_code[i : i + chunk]
            )
            normal_chunk += [normal]
            if refine_edge:
                # compute cos_angle of hit ray and surface normal
                # for edges near to the perpendicular, we dim the alpha
                normal_reg = torch.nn.functional.normalize(normal, dim=1)
                cos_angle = -(rays_d[i : i + chunk] * normal_reg).sum(-1)
                # do not affect other visible part that far from perpendicular
                mask_merged = torch.logical_and(mask, cos_angle < 0)
                alpha[mask_merged] = 0  # just set to 0 is enough
                # alpha[mask] = torch.relu(torch.tanh(cos_angle[mask] * 2))
                alpha_chunk[-1] = alpha

    d_pred = torch.cat(d_pred_chunk, 0)
    alpha = torch.cat(alpha_chunk, 0)

    ret_res = {
        "alphas": alpha.unsqueeze(1),
        "z_vals": d_pred.unsqueeze(1),
    }
    if not hit_test_only:
        ret_res["rgbs"] = torch.cat(rgb_chunk, 0).unsqueeze(1)
    if need_normal:
        ret_res["normals"] = torch.cat(normal_chunk, 0).unsqueeze(1)

    return ret_res


def render_rays_multi_neus(
    room_optimizer,
    models: Dict[str, NeuS],
    rays_list: List[torch.Tensor],
    obj_instance_ids: List[int],
    noise_std=0,
    white_back=False,
    use_sphere_tracing=True,
    refine_edge=False,
    safe_region_volume_rendering=True,
    hit_test_only=False,  # only works for NeuS
    need_normal=False,  # only works for NeuS
    render_mask=False,  # only works for NeuS
    refine_edge_obj_ids=[],
    # chunk=4096,
    chunk=99999999,  # chunk should be controlled outside
    extra_dict: Dict[str, Any] = {},
    render_kwargs: Dict[str, Any] = {},
):
    assert len(rays_list) == len(obj_instance_ids)
    if render_mask:
        assert use_sphere_tracing, "render_mask only support sphere_tracing mode"
    results = {}
    if use_sphere_tracing:
        chunk = 99999999  # sphere_tracing allows larger chunk size
    else:  # hit_test_only works only for sphere tracing mode
        hit_test_only = False
    rgbs_list = []
    alphas_list = []
    z_vals_list = []
    for i in range(len(rays_list)):
        # hack to suppress zero points
        # zero_mask = z_vals[:, -1] == 0
        # xyz_fine[zero_mask] = 0

        obj_id = obj_instance_ids[i]
        if len(refine_edge_obj_ids) > 0:
            if obj_id in refine_edge_obj_ids:
                refine_edge = True
            else:
                refine_edge = False
        rays = rays_list[i]
        N_rays = rays.shape[0]

        obj_code = extra_dict[f"embedding_inst_{obj_id}"].view(N_rays, -1)
        light_code = (
            extra_dict[f"embedding_light_{obj_id}"].view(N_rays, -1)
            if f"embedding_light_{obj_id}" in extra_dict
            else None
        )
        appearance_code = (
            extra_dict[f"embedding_appearance_{obj_id}"].view(N_rays, -1)
            if f"embedding_appearance_{obj_id}" in extra_dict
            else None
        )

        model = models[f"neus_{obj_id}"]

        rays_o = rays[:, 0:3].view(N_rays, 3)
        rays_d = rays[:, 3:6].view(N_rays, 3)
        near_bypass = rays[:, 6].view(N_rays, 1)
        far_bypass = rays[:, 7].view(N_rays, 1)

        zero_mask = (far_bypass != 0).squeeze()

        device = rays_o.device
        dtype = rays_o.dtype

        rays_o = rays_o[zero_mask]
        rays_d = rays_d[zero_mask]
        near_bypass = near_bypass[zero_mask]
        far_bypass = far_bypass[zero_mask]
        obj_code = obj_code[zero_mask]
        light_code = None if light_code is None else light_code[zero_mask]
        appearance_code = (
            None if appearance_code is None else appearance_code[zero_mask]
        )

        if rays_o.shape[0] > 0:  # if have valid rays to render
            if use_sphere_tracing:
                render_res = sphere_tracing_rendering(
                    model=model,
                    rays_o=rays_o,
                    rays_d=rays_d,
                    near=near_bypass,
                    far=far_bypass,
                    obj_code=obj_code,
                    light_code=light_code,
                    appearance_code=appearance_code,
                    hit_test_only=hit_test_only,
                    need_normal=need_normal,
                    refine_edge=False
                    if obj_id == 0
                    else refine_edge,  # do not refine edge for background
                    chunk=chunk,
                )
                z_vals = render_res["z_vals"]
                alphas = render_res["alphas"]
                if not hit_test_only:
                    rgbs = render_res["rgbs"]
                if need_normal:
                    results[f"normals_{obj_id}"] = render_res["normals"]

            else:
                if (
                    safe_region_volume_rendering
                ):  # we first use sphere tracing to get the exact distance
                    with torch.no_grad():
                        # acceletate with sphere tracing
                        render_res_sphere = sphere_tracing_rendering(
                            model=model,
                            rays_o=rays_o,
                            rays_d=rays_d,
                            near=near_bypass,
                            far=far_bypass,
                            obj_code=obj_code,
                            light_code=light_code,
                            appearance_code=appearance_code,
                            refine_edge=False,
                            hit_test_only=True,
                            need_normal=need_normal,
                            chunk=chunk,
                        )
                        # get exact depth to the surface
                        depth = render_res_sphere["z_vals"].view(-1, 1)
                        # set near/far near the surface
                        near_bypass = torch.clamp_min(depth - 0.1, 0.0)
                        far_bypass = torch.clamp_min(depth + 0.05, 0.0)
                        render_kwargs = copy.deepcopy(render_kwargs)
                        # with the correct surface, we can render with little sampling points
                        render_kwargs["N_samples"] = 8
                        render_kwargs["N_importance"] = 16
                render_res = volume_render(
                    rays_o=rays_o,
                    rays_d=rays_d,
                    model=model,
                    near_bypass=near_bypass,
                    far_bypass=far_bypass,
                    detailed_output=False,
                    obj_code=obj_code,
                    light_code=light_code,
                    appearance_code=appearance_code,
                    skip_accumulation=True,
                    **render_kwargs,
                )
                rgbs = render_res["radiances"]
                alphas = render_res["opacity"]
                z_vals = render_res["z_vals"]
        else:  # all the rays has been masked out
            rgbs = torch.empty((0, 1, 3), device=device, dtype=dtype)
            z_vals = torch.empty((0, 1), device=device, dtype=dtype)
            alphas = torch.empty((0, 1), device=device, dtype=dtype)

        # alphas: [N_rays, N_samples]
        # z_vals: [N_rays, N_samples]
        # rgbs: [N_rays, N_samples, 3]
        # recover to full size
        _, N_samples = z_vals.shape

        alphas_full = torch.zeros((N_rays, N_samples), device=device, dtype=dtype)
        alphas_full[zero_mask] = alphas
        alphas_list += [alphas_full]

        z_vals_full = torch.zeros((N_rays, N_samples), device=device, dtype=dtype)
        z_vals_full[zero_mask] = z_vals
        z_vals_list += [z_vals_full]

        if not hit_test_only:
            rgbs_full = torch.zeros((N_rays, N_samples, 3), device=device, dtype=dtype)
            rgbs_full[zero_mask] = rgbs.type(dtype)
            if f"autoexposure_{obj_id}" in extra_dict:
                exposure_param = extra_dict[f"autoexposure_{obj_id}"]
                scale = exposure_param[:3] + 1e-5
                shift = exposure_param[3:]
                rgbs_full = (rgbs_full - shift) / scale
            rgbs_list += [rgbs_full]

        # output real scale in physical world
        z_vals_list[i] = z_vals_list[i] * room_optimizer.get_scale_factor(obj_id)
        if obj_id == 0:
            z_vals_list[i] = (
                z_vals_list[i] * 1.01
            )  # avoid aliasing for thin object like carpet

    if hit_test_only:
        z_vals, _ = torch.stack(z_vals_list).min(0)
        alpha, _ = torch.stack(alphas_list).max(0)
        results["depth_fine"] = z_vals
        results["opacity_fine"] = alpha
    else:
        volume_rendering_multi_neus(
            results,
            "fine",
            z_vals_list,
            rgbs_list,
            alphas_list,
            noise_std,
            white_back,
        )
        if render_mask:
            mask_ids_list = []
            for i, obj_id in enumerate(obj_instance_ids):
                mask_ids_list += [torch.ones_like(rgbs_list[i]) * obj_id]
            results_mask = {}

            volume_rendering_multi_neus(
                results_mask,
                "fine",
                z_vals_list,
                mask_ids_list,
                alphas_list,
                noise_std,
                white_back,
            )
            results["rendered_instance_mask"] = results_mask["rgb_fine"]

    return results
