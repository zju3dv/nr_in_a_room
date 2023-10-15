import numpy as np
import numba as nb
import torch
import open3d as o3d
import copy

import sys
import os

sys.path.append(os.getcwd())  # noqa

from utils.util import read_json, read_yaml
from datasets.geo_utils import *


class BBoxRayHelper:
    def __init__(self, bbox_center, bbox_length):
        """
        bbox_center: [3]
        bbox_length: [3]
        """
        super().__init__()
        self.length_half = np.array(bbox_length) * 0.5
        self.center = np.array(bbox_center)
        self.bbox_bounds = np.array(
            [self.center - self.length_half, self.center + self.length_half]
        ).astype(np.float32)

    def get_center(self):
        return self.bbox_c

    def get_ray_bbox_intersections(
        self, rays_o_bbox, rays_d_bbox, scale_factor, bbox_enlarge=0
    ):
        bbox_bounds = copy.deepcopy(self.bbox_bounds)
        if bbox_enlarge > 0:
            # bbox_z_min_orig = bbox_bounds[0][2]
            bbox_bounds[0] -= bbox_enlarge
            bbox_bounds[1] += bbox_enlarge
            # bbox_bounds[0][2] = bbox_z_min_orig
        if isinstance(rays_o_bbox, torch.Tensor):
            rays_o_bbox = rays_o_bbox.detach().cpu().float().numpy()
            rays_d_bbox = rays_d_bbox.detach().cpu().float().numpy()
        bbox_mask, batch_near, batch_far = bbox_intersection_batch(
            bbox_bounds, rays_o_bbox, rays_d_bbox
        )
        bbox_mask, batch_near, batch_far = (
            torch.Tensor(bbox_mask).bool(),
            torch.Tensor(batch_near[..., None]),
            torch.Tensor(batch_far[..., None]),
        )
        batch_near, batch_far = batch_near / scale_factor, batch_far / scale_factor
        return bbox_mask.cuda(), batch_near.cuda(), batch_far.cuda()
