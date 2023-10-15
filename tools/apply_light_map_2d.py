import sys
import os

sys.path.append(os.getcwd())  # noqa
import ipdb
import numpy as np
import torch
import cv2
import imageio
import time

from utils.util import write_point_cloud


def cartesian_to_spherical(xyz):
    """
    https://stackoverflow.com/questions/4116658/faster-numpy-cartesian-to-spherical-coordinate-conversion
    """
    ptsnew = np.empty(xyz.shape)
    xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
    # ptsnew[:, 0] = np.sqrt(xy + xyz[:, 2] ** 2)
    ptsnew[:, 0] = np.arctan2(
        np.sqrt(xy), xyz[:, 2]
    )  # for elevation angle defined from Z-axis down
    # ptsnew[:, 0] = np.arctan2(
    #     xyz[:, 2], np.sqrt(xy)
    # )  # for elevation angle defined from XY-plane up
    ptsnew[:, 1] = np.arctan2(xyz[:, 1], xyz[:, 0])
    return ptsnew


def anorm(x, axis=None, keepdims=False):
    """Compute L2 norms along specified axes."""
    return np.sqrt((x * x).sum(axis=axis, keepdims=keepdims))


def normalize(v, axis=None, eps=1e-10):
    """L2 Normalize along specified axes."""
    return v / np.maximum(anorm(v, axis=axis, keepdims=True), eps)


def regularize_normal_by_clustering(normal, K=7):
    from sklearn.cluster import KMeans

    H, W, _ = normal.shape
    # use small size to predict cluster centers
    resize_factor = 4
    normal_small = cv2.resize(
        normal,
        dsize=(W // resize_factor, H // resize_factor),
        interpolation=cv2.INTER_NEAREST,
    )
    normal_small = normal_small.reshape(-1, 3)
    normal = normal.reshape(-1, 3)

    kmeans = KMeans(n_clusters=K, random_state=0).fit(normal_small)
    # normal = kmeans.cluster_centers_[kmeans.labels_]
    pred_labels = kmeans.predict(normal)
    normal = kmeans.cluster_centers_[pred_labels]
    normal = normal.reshape(H, W, 3)
    return normal


def compute_normal_from_depth(
    depth: np.ndarray,
    focal: float = None,
    is_panorama: bool = False,
):
    """
    Inputs:
        depth: [H, W]
    """
    from datasets.ray_utils import (
        get_ray_directions,
        get_ray_directions_equirectangular,
    )

    H, W = depth.shape
    if is_panorama:
        rays_d = get_ray_directions_equirectangular(H, W).numpy()  # [H, W, 3]
        pts_3d = rays_d * depth.reshape(H, W, 1)
    else:
        rays_d = get_ray_directions(H, W, focal).numpy()  # [H, W, 3]
        pts_3d = rays_d * depth.reshape(H, W, 1)
    vector_xplus1 = pts_3d[:, :-1, :] - pts_3d[:, 1:, :]  # (H, W-1, 3)
    vector_xplus1 = np.concatenate(
        (vector_xplus1, vector_xplus1[:, -1:, :]), axis=1
    )  # (H,W,3)
    vector_yplus1 = pts_3d[:-1, :, :] - pts_3d[1:, :, :]  # (H-1, W, 3)
    vector_yplus1 = np.concatenate(
        [vector_yplus1, vector_yplus1[-1:, :, :]], axis=0
    )  # (H,W,3)
    normal = np.cross(vector_xplus1, vector_yplus1, 2)  # (H,W,3)
    normal = -normalize(normal, axis=2)
    # normal_map[:, :, 0] *= -1
    normal[depth == 0] = 0
    if is_panorama:
        # t1 = time.time()
        normal = regularize_normal_by_clustering(normal)
        # t2 = time.time()
        # print(t2 - t1)
        normal[depth == 0] = 0
    # apply erode and dilate to remove border artifact
    kernel = np.ones((3, 3), np.float32)
    normal = cv2.erode(normal.astype(np.float32), kernel, iterations=1)
    normal = cv2.dilate(normal.astype(np.float32), kernel, iterations=1)
    normal[depth == 0] = 0
    return normal


def apply_light_map_augmentation(
    img: np.ndarray,
    pose: np.ndarray,
    light_map: np.ndarray,
    normal_map: np.ndarray = None,
    depth_map: np.ndarray = None,
    focal: float = None,
    is_panorama: bool = False,
    shrink_range: float = 1.0,
    horizontal_offset: float = 0.0,
    normalize_light_map: bool = True,
):
    """
    Inputs:
        img: [H, W, 3]
        normal_map: [H, W, 3]
        pose: [4, 3] Twc
        light_map: [Hn, Wn, 3]
        shrink_range: shrink light map range, so as to make the light change smaller
        horizontal_offset: in [0, 1], which can be used to augment light condition
    """
    if normal_map is None and depth_map is not None:
        normal_map = compute_normal_from_depth(depth_map, focal, is_panorama)
        # imageio.imwrite(
        #     "debug/computed_normal.png", ((normal_map + 1) / 2 * 255).astype(np.uint8)
        # )
    else:
        valid_mask = normal_map.sum(-1) != 0  # avoid making changes to invalid part
        normal_map = (normal_map.astype(np.float32) / 255) * 2 - 1
        normal_map[~valid_mask] = 0

    assert img.shape == normal_map.shape
    H, W, _ = img.shape

    # ipdb.set_trace()
    valid_mask = normal_map.sum(-1) != 0
    # full_ind = np.arange(H * W).reshape(-1)
    valid_normal = normal_map[valid_mask].reshape(-1, 3)
    valid_normal = normalize(valid_normal, axis=1)
    # no need to fix rotation of poses, normal has already in the form of NDC
    valid_normal = (pose[:3, :3] @ valid_normal.T).T  # only need rotation

    # dump to debug
    # valid_normal = np.random.randn(*valid_normal.shape) # for debug
    # valid_normal += np.random.randn(*valid_normal.shape) * 0.01
    # write_point_cloud(valid_normal, "debug/normal.ply")

    spherical_coords = cartesian_to_spherical(valid_normal)
    Hn, Wn, _ = light_map.shape
    theta, phi = spherical_coords[:, 0], spherical_coords[:, 1]

    # [0, 1]
    px = (phi + np.pi) / (np.pi * 2)
    py = theta / np.pi

    # apply horizontal offset
    assert horizontal_offset >= 0 and horizontal_offset <= 1
    px += horizontal_offset
    px[px > 1] = px[px > 1] - 1

    # assert (px < 0).sum() == 0 and (px > 1).sum() == 0
    # assert (py < 0).sum() == 0 and (py > 1).sum() == 0

    # to [-1, 1]
    coords_xy = np.stack([px, py]) * 2 - 1

    # debug in coord map
    # coords_map = np.ones((Hn, Wn))
    # coords_map[(py * Hn).astype(int), (px * Wn).astype(int)] = 0
    # imageio.imwrite("debug/test_light_map/dump_coord_map.png", coords_map)

    # bilinear sample from images
    light_map = (
        torch.from_numpy(light_map).float().permute(2, 0, 1).unsqueeze(0)
    )  # [1, 3, H, W]

    if normalize_light_map:
        mean_light = light_map.view(3, -1).mean(axis=1, keepdim=False)
        light_map = light_map - mean_light.view(1, 3, 1, 1) + 1.0

    coords_xy = (
        torch.from_numpy(coords_xy).float().permute(1, 0).unsqueeze(0).unsqueeze(0)
    )  # [1, 1, N, 2]
    sampled_light = torch.nn.functional.grid_sample(
        light_map, coords_xy, mode="bilinear", padding_mode="border", align_corners=True
    )  # [1, 3, 1, N]
    sampled_light = sampled_light.view(3, -1).permute(1, 0).numpy()  # [N, 3]
    sampled_light = (sampled_light - 1) * shrink_range + 1
    img = img.copy().astype(np.float32)  # avoid overflow
    img[valid_mask] = img[valid_mask] * sampled_light
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def test_perspective():
    # light_map = imageio.imread("debug/test_light_map/light_map.exr")
    light_map = cv2.imread(os.path.join("debug/test_light_map/light_map.exr"), -1)
    light_map = cv2.cvtColor(light_map, cv2.COLOR_BGR2RGB)
    normal_map = imageio.imread("debug/test_light_map/00015.normal.png")
    # depth_map = cv2.imread("debug/test_light_map/00015.depth.png", flags=-1)
    # depth_map = cv2.imread("debug/test_light_map/00015.depth.hdr", flags=-1)[:, :, 0]
    depth_map = np.load("debug/test_light_map/00015.depth.npy")
    # ipdb.set_trace()
    # depth_map = depth_map * 5e-3
    orig_image = imageio.imread("debug/test_light_map/00015.png")

    pose = np.array(
        [
            [
                -0.8713186979293823,
                0.3679307997226715,
                -0.32470086216926575,
                -0.4804389476776123,
            ],
            [
                -0.4907175600528717,
                -0.6532983779907227,
                0.5765392780303955,
                0.8530679941177368,
            ],
            [
                -1.0880500056487108e-08,
                0.6616858243942261,
                0.7497812509536743,
                1.1094030141830444,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ).reshape(4, 4)

    # # use given normal map
    # img = apply_light_map_augmentation(
    #     img=orig_image,
    #     pose=pose,
    #     light_map=light_map,
    #     normal_map=normal_map,
    #     shrink_range=0.8,
    #     horizontal_offset=0.5,
    # )
    fov_x = 1.3962634015954636
    h, w = depth_map.shape
    focal = 0.5 * w / np.tan(0.5 * fov_x)  # original focal length
    print(focal)
    # compute normal map from depth
    img = apply_light_map_augmentation(
        img=orig_image,
        pose=pose,
        light_map=light_map,
        depth_map=depth_map,
        focal=focal,
        shrink_range=0.8,
        horizontal_offset=0.1,
    )

    imageio.imwrite("debug/test_light_map/applied.png", img)


def test_panorama():
    light_map = cv2.imread(os.path.join("debug/test_light_map_pano/light_map.exr"), -1)
    light_map = cv2.cvtColor(light_map, cv2.COLOR_BGR2RGB)
    # depth_map = np.load("debug/test_light_map_pano/depth.npy")
    depth_map = np.load("debug/test_light_map_pano/depth.npy")
    # depth_map = cv2.imread("debug/test_light_map_pano/depth.png", flags=-1)
    # depth_map = (depth_map >> 3) | (depth_map << 13)  # for ig background
    # depth_map = depth_map * 1e-3
    # ipdb.set_trace()
    # depth_map = depth_map * 5e-3
    orig_image = imageio.imread("debug/test_light_map_pano/rgb.png")
    normal_map = imageio.imread("debug/test_light_map_pano/normal.png")

    pose = np.array(
        [
            [1.0, 0.0, 0.0, -5.099999904632568],
            [0.0, 0.0, 1.0, -3.200000047683716],
            [-0.0, -1.0, -0.0, 1.600000023841858],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ).reshape(4, 4)

    # compute normal map from depth
    img = apply_light_map_augmentation(
        img=orig_image,
        pose=pose,
        light_map=light_map,
        depth_map=depth_map,
        # normal_map=normal_map,
        focal=None,
        is_panorama=True,
        shrink_range=0.5,
        horizontal_offset=0.1,
    )

    imageio.imwrite("debug/test_light_map_pano/applied.png", img)


if __name__ == "__main__":
    # test_perspective()
    test_panorama()
