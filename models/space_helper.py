import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import itertools
from tqdm import tqdm
from einops import rearrange, reduce, repeat

# from torch_cluster import knn
import time

import sys
import os

from yaml.nodes import ScalarNode

sys.path.append(os.getcwd())  # noqa
from utils.util import *
from models.nerf import Embedding
import open3d as o3d


def dump_voxel_occupancy_map(
    voxel_occupancy_map, voxel_size, scale_factor, scene_center
):
    idx_occu = torch.nonzero(voxel_occupancy_map)
    voxel_xyz = (
        idx_occu.float().detach().cpu().numpy() * voxel_size * scale_factor
        + scene_center
    )
    write_point_cloud(voxel_xyz, "voxel.ply")


def randomly_set_occupancy_mask_to_true(occupancy_mask, ratio=0.1):
    # which may help with empty embedding learning?
    assert len(occupancy_mask.shape) == 1
    N = occupancy_mask.shape
    idx_empty = torch.nonzero(occupancy_mask == False)
    if idx_empty.shape[0] < 10:
        return occupancy_mask
    # randomly set occupancy which has been marked as False to True
    N_empty = len(idx_empty)
    idx_rand_choice = torch.randperm(N_empty)[: int(N_empty * ratio)]
    occupancy_mask[idx_empty[idx_rand_choice]] = True
    return occupancy_mask


def marker_border_surface(occupancy, thickness=1):
    # marker border surface so that it can learn far textures in border surface
    occupancy[:thickness, :, :] = True
    occupancy[-thickness:, :, :] = True
    occupancy[:, :thickness, :] = True
    occupancy[:, -thickness:, :] = True
    occupancy[:, :, :thickness] = True
    occupancy[:, :, -thickness] = True
    return occupancy


# def offset_points(point_xyz, quarter_voxel=1, offset_only=False, bits=2):
#     c = torch.arange(1, 2 * bits, 2, device=point_xyz.device)
#     ox, oy, oz = torch.meshgrid([c, c, c])
#     offset = (torch.cat([
#                     ox.reshape(-1, 1),
#                     oy.reshape(-1, 1),
#                     oz.reshape(-1, 1)], 1).type_as(point_xyz) - bits) / float(bits - 1)
#     if not offset_only:
#         return point_xyz.unsqueeze(1) + offset.unsqueeze(0).type_as(point_xyz) * quarter_voxel
#     return offset.type_as(point_xyz) * quarter_voxel


# def splitting_points(point_xyz, point_xyz_quantize, half_voxel):
#     # generate new centers
#     quarter_voxel = half_voxel * .5
#     new_points = offset_points(point_xyz + half_voxel, quarter_voxel).reshape(-1, 3)
#     new_coords = offset_points(point_xyz_quantize * 2).reshape(-1, 3).round().long()
#     pseudo_max_length = 1e5
#     unique_keys = new_coords[:,0] * pseudo_max_length ** 2 + new_coords[:,1] * pseudo_max_length + new_coords[:,2]
#     # get unique keys and inverse indices (for original key0, where it maps to in keys)
#     unique_keys, unique_idx = torch.unique(unique_keys, dim=0, sorted=True, return_inverse=True)
#     new_points = new_points[unique_idx]
#     new_coords = new_coords[unique_idx]
#     return new_points, new_coords
