import torch
from einops import rearrange, reduce, repeat

# __all__ = ['render_rays']


def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.
    Inputs:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
        weights: (N_rays, N_samples_)
        N_importance: the number of samples to draw from the distribution
        det: deterministic or not
        eps: a small number to prevent division by zero
    Outputs:
        samples: the sampled samples
    """
    N_rays, N_samples_ = weights.shape
    weights = weights + eps  # prevent division by zero (don't do inplace op!)
    pdf = weights / reduce(weights, "n1 n2 -> n1 1", "sum")  # (N_rays, N_samples_)
    cdf = torch.cumsum(pdf, -1)  # (N_rays, N_samples), cumulative distribution function
    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], -1)  # (N_rays, N_samples_+1)
    # padded to 0~1 inclusive

    if det:
        u = torch.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        u = torch.rand(N_rays, N_importance, device=bins.device)
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp_min(inds - 1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = rearrange(
        torch.stack([below, above], -1), "n1 n2 c -> n1 (n2 c)", c=2
    )
    cdf_g = rearrange(torch.gather(cdf, 1, inds_sampled), "n1 (n2 c) -> n1 n2 c", c=2)
    bins_g = rearrange(torch.gather(bins, 1, inds_sampled), "n1 (n2 c) -> n1 n2 c", c=2)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom[denom < eps] = 1  # denom equals 0 means a bin has weight 0,
    # in which case it will not be sampled
    # anyway, therefore any value for it is fine (set to 1 here)

    samples = bins_g[..., 0] + (u - cdf_g[..., 0]) / denom * (
        bins_g[..., 1] - bins_g[..., 0]
    )
    return samples


def render_rays(
    models,
    embeddings,
    rays,
    N_samples=64,
    use_disp=False,
    perturb=0,
    noise_std=1,
    N_importance=0,
    chunk=1024 * 32,
    white_back=False,
    test_time=False,
    **kwargs,
):
    """
    Render rays by computing the output of @model applied on @rays
    Inputs:
        models: list of NeRF models (coarse and fine) defined in nerf.py
        embeddings: list of embedding models of origin and direction defined in nerf.py
        rays: (N_rays, 3+3+2), ray origins and directions, near and far depths
        N_samples: number of coarse samples per ray
        use_disp: whether to sample in disparity space (inverse depth)
        perturb: factor to perturb the sampling position on the ray (for coarse model only)
        noise_std: factor to perturb the model's prediction of sigma
        N_importance: number of fine samples per ray
        chunk: the chunk size in batched inference
        white_back: whether the background is white (dataset dependent)
        test_time: whether it is test (inference only) or not. If True, it will not do inference
                   on coarse rgb to save time
    Outputs:
        result: dictionary containing final rgb and depth maps for coarse and fine models
    """

    def inference(results, model, typ, xyz, z_vals, test_time=False, **kwargs):
        """
        Helper function that performs model inference.
        Inputs:
            results: a dict storing all results
            model: NeRF model (coarse or fine)
            typ: 'coarse' or 'fine'
            xyz: (N_rays, N_samples_, 3) sampled positions
                  N_samples_ is the number of sampled points in each ray;
                             = N_samples for coarse model
                             = N_samples+N_importance for fine model
            z_vals: (N_rays, N_samples_) depths of the sampled positions
            test_time: test time or not
        Outputs:
            if weights_only:
                weights: (N_rays, N_samples_): weights of each sample
            else:
                rgb_final: (N_rays, 3) the final rgb image
                depth_final: (N_rays) depth map
                weights: (N_rays, N_samples_): weights of each sample
        """
        N_samples_ = xyz.shape[1]
        xyz_ = rearrange(xyz, "n1 n2 c -> (n1 n2) c")  # (N_rays*N_samples_, 3)

        # Perform model inference to get rgb and raw sigma
        B = xyz_.shape[0]
        out_chunks = []

        # for instance mask
        compute_3d_mask = kwargs.get("compute_3d_mask", False)
        is_eval = kwargs.get("is_eval", False)

        mask_3d_instance_chunk = []
        instance_rgb_chunk = []
        if typ == "coarse" and test_time and "fine" in models:
            inst_embedded_ = repeat(
                kwargs["embedding_inst"], "n1 c -> (n1 n2) c", n2=N_samples_
            )
            for i in range(0, B, chunk):
                xyz_embedded, inst_voxel_embedded = embedding_xyz(xyz_[i : i + chunk])
                out_sigma = model(xyz_embedded, sigma_only=True)
                out_chunks += [out_sigma]
                if compute_3d_mask:
                    input_dict = {
                        "xyz_embedded": xyz_embedded,
                        "inst_voxel_ftr": inst_voxel_embedded,
                        "inst_embedded": inst_embedded_[i : i + chunk],
                    }
                    assert False, "Not implemented"
                    mask_3d_instance_chunk += [model.forward_instance_mask(input_dict)]
            out = torch.cat(out_chunks, 0)
            sigmas = rearrange(out, "(n1 n2) 1 -> n1 n2", n1=N_rays, n2=N_samples_)
            if compute_3d_mask:
                mask_3d_instance = torch.cat(mask_3d_instance_chunk, 0)
                mask_3d_instance = rearrange(
                    mask_3d_instance, "(n1 n2) 1 -> n1 n2", n1=N_rays, n2=N_samples_
                )
        else:  # infer rgb and sigma and others
            dir_embedded_ = repeat(dir_embedded, "n1 c -> (n1 n2) c", n2=N_samples_)
            # (N_rays*N_samples_, embed_dir_channels)
            # a_embedded_ = repeat(kwargs['embedding_a'], 'n1 c -> (n1 n2) c', n2=N_samples_)
            inst_embedded_ = repeat(
                kwargs["embedding_inst"], "n1 c -> (n1 n2) c", n2=N_samples_
            )
            # assert dir_embedded_.shape[0] == a_embedded_.shape[0]
            assert dir_embedded_.shape[0] == inst_embedded_.shape[0]
            for i in range(0, B, chunk):
                xyz_embedded = embedding_xyz(xyz_[i : i + chunk])

                xyzdir_embedded = torch.cat(
                    [xyz_embedded, dir_embedded_[i : i + chunk]], 1
                )
                out_sigma_color, _ = model(xyzdir_embedded, sigma_only=False)
                out_chunks += [out_sigma_color]
                if compute_3d_mask:
                    input_dict = {
                        "xyz_embedded": xyz_embedded,
                        "inst_embedded": inst_embedded_[i : i + chunk],
                        # 'embedding_a': a_embedded_[i:i+chunk],
                        "input_dir": dir_embedded_[i : i + chunk],
                    }
                    inst_sigma, inst_rgb = model.forward_instance_mask(input_dict)
                    mask_3d_instance_chunk += [inst_sigma]
                    instance_rgb_chunk += [inst_rgb]

            out = torch.cat(out_chunks, 0)
            # out = out.view(N_rays, N_samples_, 4)
            out = rearrange(out, "(n1 n2) c -> n1 n2 c", n1=N_rays, n2=N_samples_, c=4)
            rgbs = out[..., :3]  # (N_rays, N_samples_, 3)
            sigmas = out[..., 3]  # (N_rays, N_samples_)
            if compute_3d_mask:
                mask_3d_instance = torch.cat(mask_3d_instance_chunk, 0)
                mask_3d_instance = rearrange(
                    mask_3d_instance, "(n1 n2) 1 -> n1 n2", n1=N_rays, n2=N_samples_
                )
                instance_rgb = torch.cat(instance_rgb_chunk, 0)
                instance_rgb = rearrange(
                    instance_rgb, "(n1 n2) c -> n1 n2 c", n1=N_rays, n2=N_samples_
                )

        # hack to suppress zero values
        zero_mask = z_vals[:, -1] == 0
        sigmas[zero_mask] = 0
        if compute_3d_mask:
            mask_3d_instance[zero_mask] = 0

        # Convert these values using volume rendering (Section 4)
        deltas = z_vals[:, 1:] - z_vals[:, :-1]  # (N_rays, N_samples_-1)
        delta_inf = 1e10 * torch.ones_like(
            deltas[:, :1]
        )  # (N_rays, 1) the last delta is infinity
        deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)

        # compute alpha by the formula (3)
        noise = torch.randn_like(sigmas) * noise_std
        noise[zero_mask] = 0
        alphas = 1 - torch.exp(
            -deltas * torch.relu(sigmas + noise)
        )  # (N_rays, N_samples_)

        alphas_shifted = torch.cat(
            [torch.ones_like(alphas[:, :1]), 1 - alphas + 1e-10], -1
        )  # [1, 1-a1, 1-a2, ...]
        weights = alphas * torch.cumprod(
            alphas_shifted[:, :-1], -1
        )  # (N_rays, N_samples_)

        weights_sum = reduce(
            weights, "n1 n2 -> n1", "sum"
        )  # (N_rays), the accumulated opacity along the rays
        # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically

        results[f"weights_{typ}"] = weights
        results[f"opacity_{typ}"] = weights_sum
        results[f"z_vals_{typ}"] = z_vals
        if test_time and typ == "coarse" and "fine" in models:
            return

        rgb_map = reduce(
            rearrange(weights, "n1 n2 -> n1 n2 1") * rgbs, "n1 n2 c -> n1 c", "sum"
        )
        depth_map = reduce(weights * z_vals, "n1 n2 -> n1", "sum")

        if white_back:
            rgb_map = rgb_map + 1 - weights_sum.unsqueeze(-1)

        results[f"rgb_{typ}"] = rgb_map
        results[f"depth_{typ}"] = depth_map

        # if is_eval and compute_3d_mask:
        if compute_3d_mask:
            # apply to sigma
            # sigmas_masked = sigmas.detach() * mask_3d_instance
            # TODO(ybbbbt): directly learn instance sigma seems better than learning a mask
            sigmas_masked = mask_3d_instance
            noise_masked = torch.randn_like(sigmas_masked) * noise_std
            noise_masked[zero_mask] = 0
            alphas_masked = 1 - torch.exp(
                -deltas * torch.relu(sigmas_masked + noise_masked)
            )  # (N_rays, N_samples_)
            # mask out occlusion
            # if not is_eval:
            #     # mask true -> area farther than the rendered scene depth
            #     bound_th = kwargs['frustum_bound']
            #     occlusion_mask = (results[f'depth_{typ}'].unsqueeze(1).expand(N_rays, N_samples_) + bound_th) < z_vals
            #     if 'instance_mask' in kwargs:
            #         # for visible instance area, we do not suppress the alphas
            #         occlusion_mask[kwargs['instance_mask'].expand(N_rays, N_samples_)] = False
            #     # mask out object alphas farther than the rendered scene depth
            #     alphas_masked[occlusion_mask] = 0
            alphas_shifted_masked = torch.cat(
                [torch.ones_like(alphas_masked[:, :1]), 1 - alphas_masked + 1e-10], -1
            )  # [1, 1-a1, 1-a2, ...]
            weights_instance = alphas_masked * torch.cumprod(
                alphas_shifted_masked[:, :-1], -1
            )  # (N_rays, N_samples_)

            # directly apply to weights
            # weights_instance = weights * mask_3d_instance

            weights_sum_instance = reduce(weights_instance, "n1 n2 -> n1", "sum")

            # compute instance rgb and depth
            rgb_instance_map = reduce(
                rearrange(weights_instance, "n1 n2 -> n1 n2 1") * instance_rgb,
                "n1 n2 c -> n1 c",
                "sum",
            )
            # weights_instance, 'n1 n2 -> n1 n2 1')*rgbs.detach(), 'n1 n2 c -> n1 c', 'sum')
            depth_instance_map = reduce(weights_instance * z_vals, "n1 n2 -> n1", "sum")
            if white_back:
                rgb_instance_map = (
                    rgb_instance_map + 1 - weights_sum_instance.unsqueeze(-1)
                )
            results[f"rgb_instance_{typ}"] = rgb_instance_map
            results[f"depth_instance_{typ}"] = depth_instance_map
            results[f"opacity_instance_{typ}"] = weights_sum_instance

            if kwargs.get("rays_in_bbox", False):  # for pdf sampling
                # print(f'eval overwrite weights_{typ}')
                results[f"weights_{typ}"] = weights_instance
        return

    embedding_xyz, embedding_dir = embeddings["xyz"], embeddings["dir"]

    # Decompose the inputs
    N_rays = rays.shape[0]
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]  # both (N_rays, 3)
    near, far = rays[:, 6:7], rays[:, 7:8]  # both (N_rays, 1)
    # Embed direction
    dir_embedded = embedding_dir(
        kwargs.get("view_dir", rays_d)
    )  # (N_rays, embed_dir_channels)

    rays_o = rearrange(rays_o, "n1 c -> n1 1 c")
    rays_d = rearrange(rays_d, "n1 c -> n1 1 c")

    # compute intersection to update near and far
    # near, far = embedding_xyz.ray_box_intersection(rays_o, rays_d, near, far)

    # Sample depth points
    z_steps = torch.linspace(0, 1, N_samples, device=rays.device)  # (N_samples)
    if not use_disp:  # use linear sampling in depth space
        z_vals = near * (1 - z_steps) + far * z_steps
    else:  # use linear sampling in disparity space
        z_vals = 1 / (1 / near * (1 - z_steps) + 1 / far * z_steps)

    z_vals = z_vals.expand(N_rays, N_samples)

    if perturb > 0:  # perturb sampling depths (z_vals)
        z_vals_mid = 0.5 * (
            z_vals[:, :-1] + z_vals[:, 1:]
        )  # (N_rays, N_samples-1) interval mid points
        # get intervals between samples
        upper = torch.cat([z_vals_mid, z_vals[:, -1:]], -1)
        lower = torch.cat([z_vals[:, :1], z_vals_mid], -1)

        perturb_rand = perturb * torch.rand_like(z_vals)
        z_vals = lower + (upper - lower) * perturb_rand

    xyz_coarse = rays_o + rays_d * rearrange(z_vals, "n1 n2 -> n1 n2 1")

    # hack to suppress zero points
    zero_mask = z_vals[:, -1] == 0
    xyz_coarse[zero_mask] = 0

    results = {}
    inference(
        results, models["coarse"], "coarse", xyz_coarse, z_vals, test_time, **kwargs
    )

    if N_importance > 0:  # sample points for fine model
        z_vals_mid = 0.5 * (
            z_vals[:, :-1] + z_vals[:, 1:]
        )  # (N_rays, N_samples-1) interval mid points
        z_vals_ = sample_pdf(
            z_vals_mid,
            results["weights_coarse"][:, 1:-1].detach(),
            N_importance,
            det=(perturb == 0),
        )
        # detach so that grad doesn't propogate to weights_coarse from here

        z_vals = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)[0]
        # combine coarse and fine samples

        xyz_fine = rays_o + rays_d * rearrange(z_vals, "n1 n2 -> n1 n2 1")
        # hack to suppress zero points
        xyz_fine[zero_mask] = 0

        inference(
            results, models["fine"], "fine", xyz_fine, z_vals, test_time, **kwargs
        )

    return results
