import torch
from einops import rearrange, reduce, repeat

import sys
import os

sys.path.append(os.getcwd())  # noqa

from models.rendering import sample_pdf


def inference_from_model(
    model, embedding_xyz, dir_embedded, xyz, z_vals, chunk, instance_id, kwargs={}
):
    N_rays = xyz.shape[0]
    N_samples_ = xyz.shape[1]
    xyz_ = rearrange(xyz, "n1 n2 c -> (n1 n2) c")  # (N_rays*N_samples_, 3)

    # Perform model inference to get rgb and raw sigma
    B = xyz_.shape[0]
    mask_3d_instance_chunk = []
    instance_rgb_chunk = []

    # hack to suppress zero values
    zero_mask = z_vals[:, -1] == 0
    zero_mask_repeat = repeat(zero_mask, "n1 -> (n1 n2)", n2=N_samples_)

    dir_embedded_ = repeat(dir_embedded, "n1 c -> (n1 n2) c", n2=N_samples_)
    # (N_rays*N_samples_, embed_dir_channels)
    # a_embedded_ = repeat(kwargs['embedding_a'], 'n1 c -> (n1 n2) c', n2=N_samples_)
    inst_embedded_ = repeat(
        kwargs["embedding_inst_{}".format(instance_id)],
        "n1 c -> (n1 n2) c",
        n2=N_samples_,
    )
    assert dir_embedded_.shape[0] == inst_embedded_.shape[0]
    # assert dir_embedded_.shape[0] == a_embedded_.shape[0]

    for i in range(0, B, chunk):
        # xyz_embedded, inst_voxel_embedded = embedding_xyz(xyz_[i:i+chunk])
        xyz_embedded = embedding_xyz(xyz_[i : i + chunk])
        xyz_embedded[zero_mask_repeat[i : i + chunk], :] = 0

        input_dict = {
            "xyz_embedded": xyz_embedded,
            # 'inst_voxel_ftr': inst_voxel_embedded,
            "inst_embedded": inst_embedded_[i : i + chunk],
            # 'embedding_a': a_embedded_[i:i+chunk],
            "input_dir": dir_embedded_[i : i + chunk],
        }
        # inst_sigma, inst_rgb = model.forward_instance_mask(input_dict)
        inst_sigma, inst_rgb = model.forward_instance_mask_skip_empty(input_dict)
        mask_3d_instance_chunk += [inst_sigma]
        instance_rgb_chunk += [inst_rgb]

    mask_3d_instance = torch.cat(mask_3d_instance_chunk, 0)
    mask_3d_instance = rearrange(
        mask_3d_instance, "(n1 n2) 1 -> n1 n2", n1=N_rays, n2=N_samples_
    )
    instance_rgb = torch.cat(instance_rgb_chunk, 0)
    instance_rgb = rearrange(
        instance_rgb, "(n1 n2) c -> n1 n2 c", n1=N_rays, n2=N_samples_
    )
    mask_3d_instance[zero_mask] = 0
    instance_rgb[zero_mask] = 0
    return instance_rgb, mask_3d_instance


def volume_rendering_multi(
    results,
    typ,
    z_vals_list,
    rgbs_list,
    sigmas_list,
    noise_std,
    white_back,
    obj_ids_list=None,
):
    N_objs = len(z_vals_list)
    # order via z_vals
    z_vals = torch.cat(z_vals_list, 1)  # (N_rays, N_samples*N_objs)
    rgbs = torch.cat(rgbs_list, 1)  # (N_rays, N_samples*N_objs, 3)
    sigmas = torch.cat(sigmas_list, 1)  # (N_rays, N_samples*N_objs)

    z_vals, idx_sorted = torch.sort(z_vals, -1)
    for i in range(3):
        rgbs[:, :, i] = torch.gather(rgbs[:, :, i].clone(), dim=1, index=idx_sorted)
    sigmas = torch.gather(sigmas, dim=1, index=idx_sorted)
    # record object ids for recovering weights of each object after sorting
    if obj_ids_list != None:
        obj_ids = torch.cat(obj_ids_list, -1)
        results[f"obj_ids_{typ}"] = torch.gather(obj_ids, dim=1, index=idx_sorted)

    # Convert these values using volume rendering (Section 4)
    deltas = z_vals[:, 1:] - z_vals[:, :-1]  # (N_rays, N_samples_-1)
    # delta_inf = 1e10 * torch.ones_like(deltas[:, :1]) # (N_rays, 1) the last delta is infinity
    delta_inf = torch.zeros_like(
        deltas[:, :1]
    )  # (N_rays, 1) the last delta is infinity
    deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)

    # compute alpha by the formula (3)
    noise = torch.randn_like(sigmas) * noise_std
    alphas = 1 - torch.exp(-deltas * torch.relu(sigmas + noise))  # (N_rays, N_samples_)

    alphas_shifted = torch.cat(
        [torch.ones_like(alphas[:, :1]), 1 - alphas + 1e-10], -1
    )  # [1, 1-a1, 1-a2, ...]
    weights = alphas * torch.cumprod(alphas_shifted[:, :-1], -1)  # (N_rays, N_samples_)

    weights_sum = reduce(
        weights, "n1 n2 -> n1", "sum"
    )  # (N_rays), the accumulated opacity along the rays
    # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically

    results[f"weights_{typ}"] = weights
    results[f"opacity_{typ}"] = weights_sum
    results[f"z_vals_{typ}"] = z_vals

    rgb_map = reduce(
        rearrange(weights, "n1 n2 -> n1 n2 1") * rgbs, "n1 n2 c -> n1 c", "sum"
    )
    depth_map = reduce(weights * z_vals, "n1 n2 -> n1", "sum")

    if white_back:
        rgb_map = rgb_map + 1 - weights_sum.unsqueeze(-1)

    results[f"rgb_{typ}"] = rgb_map
    results[f"depth_{typ}"] = depth_map


def render_rays_multi(
    models,
    embeddings,
    rays_list,
    obj_instance_ids,
    N_samples=64,
    use_disp=False,
    perturb=0,
    noise_std=0,
    N_importance=0,
    chunk=1024 * 32,
    white_back=False,
    individual_weight_for_coarse=True,
    obj_bg_relative_scale=1,
    **kwargs,
):
    """
    individual_weight_for_coarse: individual coarse sampling for each
    """

    embedding_xyz, embedding_dir = embeddings["xyz"], embeddings["dir"]

    assert len(rays_list) == len(obj_instance_ids)

    z_vals_list = []
    xyz_coarse_list = []
    dir_embedded_list = []
    rays_o_list = []
    rays_d_list = []

    for idx, rays in enumerate(rays_list):
        # Decompose the inputs
        N_rays = rays.shape[0]
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]  # both (N_rays, 3)
        near, far = rays[:, 6:7], rays[:, 7:8]  # both (N_rays, 1)

        # Embed direction
        dir_embedded = embedding_dir(rays_d)  # (N_rays, embed_dir_channels)

        rays_o = rearrange(rays_o, "n1 c -> n1 1 c")
        rays_d = rearrange(rays_d, "n1 c -> n1 1 c")

        # compute intersection to update near and far
        # near, far = embedding_xyz.ray_box_intersection(rays_o, rays_d, near, far)

        rays_o_list += [rays_o]
        rays_d_list += [rays_d]

        # Sample depth points
        z_steps = torch.linspace(0, 1, N_samples, device=rays.device)  # (N_samples)
        if not use_disp:  # use linear sampling in depth space
            z_vals = near * (1 - z_steps) + far * z_steps
        else:  # use linear sampling in disparity space
            z_vals = 1 / (1 / near * (1 - z_steps) + 1 / far * z_steps)

        z_vals = z_vals.expand(N_rays, N_samples)

        # assert perturb > 0 # to avoid same z_vals for different objects
        if perturb > 0:  # perturb sampling depths (z_vals)
            z_vals_mid = 0.5 * (
                z_vals[:, :-1] + z_vals[:, 1:]
            )  # (N_rays, N_samples-1) interval mid points
            # get intervals between samples
            upper = torch.cat([z_vals_mid, z_vals[:, -1:]], -1)
            lower = torch.cat([z_vals[:, :1], z_vals_mid], -1)

            perturb_rand = perturb * torch.rand_like(z_vals)
            z_vals = lower + (upper - lower) * perturb_rand

        # if we have ray mask (e.g. bbox), we clip z values
        if rays.shape[1] == 10:
            bbox_mask_near, bbox_mask_far = rays[:, 8:9], rays[:, 9:10]
            z_val_mask = torch.logical_and(
                z_vals > bbox_mask_near, z_vals < bbox_mask_far
            )
            z_vals[z_val_mask] = bbox_mask_far.repeat(1, z_vals.shape[1])[z_val_mask]

        xyz_coarse = rays_o + rays_d * rearrange(z_vals, "n1 n2 -> n1 n2 1")

        # hack to suppress zero points
        zero_mask = z_vals[:, -1] == 0
        xyz_coarse[zero_mask] = 0

        # save for each rays batch
        # print(xyz_coarse.shape)
        xyz_coarse_list += [xyz_coarse]
        z_vals_list += [z_vals]
        dir_embedded_list += [dir_embedded]

    # inference for each objects
    rgbs_list = []
    sigmas_list = []
    obj_ids_list = []
    for i in range(len(rays_list)):
        model_suffix = "_bg" if obj_instance_ids[i] == 0 else ""
        rgbs, sigmas = inference_from_model(
            models[f"coarse{model_suffix}"],
            embedding_xyz,
            dir_embedded_list[i],
            xyz_coarse_list[i],
            z_vals_list[i],
            chunk,
            obj_instance_ids[i],
            kwargs,
        )
        rgbs_list += [rgbs]
        sigmas_list += [sigmas]
        obj_ids_list += [torch.ones_like(sigmas) * i]

    results = {}
    if individual_weight_for_coarse:
        for i in range(len(rays_list)):
            results_local = {}
            volume_rendering_multi(
                results_local,
                "coarse",
                z_vals_list[i : i + 1],
                rgbs_list[i : i + 1],
                sigmas_list[i : i + 1],
                noise_std,
                white_back,
                obj_ids_list[i : i + 1],
            )
            results[f"weights_coarse_{i}"] = results_local["weights_coarse"]
    else:
        volume_rendering_multi(
            results,
            "coarse",
            z_vals_list,
            rgbs_list,
            sigmas_list,
            noise_std,
            white_back,
            obj_ids_list,
        )

    if N_importance > 0:  # sample points for fine model
        rgbs_list = []
        sigmas_list = []
        z_vals_fine_list = []
        for i in range(len(rays_list)):
            z_vals = z_vals_list[i]
            # hack to suppress zero points
            zero_mask = z_vals[:, -1] == 0
            z_vals_mid = 0.5 * (
                z_vals[:, :-1] + z_vals[:, 1:]
            )  # (N_rays, N_samples-1) interval mid points
            # recover weights according to z_vals from results
            if individual_weight_for_coarse:
                weights_ = results[f"weights_coarse_{i}"]
            else:
                weights_ = rearrange(
                    weights_, "(n1 n2) -> n1 n2", n1=N_rays, n2=N_samples
                )
                assert weights_.numel() == N_rays * N_samples
                weights_ = results["weights_coarse"][results["obj_ids_coarse"] == i]
            z_vals_ = sample_pdf(
                z_vals_mid, weights_[:, 1:-1].detach(), N_importance, det=(perturb == 0)
            )
            # detach so that grad doesn't propogate to weights_coarse from here

            z_vals = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)[0]

            # if we have ray mask (e.g. bbox), we clip z values
            rays = rays_list[i]
            if rays.shape[1] == 10:
                bbox_mask_near, bbox_mask_far = rays[:, 8:9], rays[:, 9:10]
                z_val_mask = torch.logical_and(
                    z_vals > bbox_mask_near, z_vals < bbox_mask_far
                )
                z_vals[z_val_mask] = bbox_mask_far.repeat(1, z_vals.shape[1])[
                    z_val_mask
                ]

                # combine coarse and fine samples
            z_vals_fine_list += [z_vals]

            xyz_fine = rays_o_list[i] + rays_d_list[i] * rearrange(
                z_vals, "n1 n2 -> n1 n2 1"
            )

            xyz_fine[zero_mask] = 0

            model_suffix = "_bg" if obj_instance_ids[i] == 0 else ""
            rgbs, sigmas = inference_from_model(
                models[f"fine{model_suffix}"],
                embedding_xyz,
                dir_embedded_list[i],
                xyz_fine,
                z_vals_fine_list[i],
                chunk,
                obj_instance_ids[i],
                kwargs,
            )
            # make sure that object and bg in the same scale
            if obj_instance_ids[i] == 0:
                z_vals_fine_list[i] = z_vals_fine_list[i] * obj_bg_relative_scale
                # TODO(ybbbbt): actually this is not so correct?
                sigmas = sigmas / obj_bg_relative_scale
            rgbs_list += [rgbs]
            sigmas_list += [sigmas]

        volume_rendering_multi(
            results,
            "fine",
            z_vals_fine_list,
            rgbs_list,
            sigmas_list,
            noise_std,
            white_back,
        )
    return results
