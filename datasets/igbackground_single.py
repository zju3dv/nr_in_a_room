import sys

sys.path.append(".")

import torch
from torch.utils import data
from torch.utils.data import Dataset
import json
import numpy as np
import os
import cv2
import itertools
import imageio
import random
import copy
from collections import defaultdict
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm

from datasets.ray_utils import *
from datasets.geo_utils import *
from datasets.image_utils import *
from utils.util import list_dir, read_json, read_yaml, remove_stdout_line, write_idx
from tools.apply_light_map_2d import apply_light_map_augmentation


class IGBackground_Single(Dataset):
    def __init__(self, batch_size, split="train", img_wh=(640, 480), config=None):
        self.split = split
        # assert img_wh[0] == img_wh[1], 'image width must equal image height!'
        self.img_wh = img_wh
        self.define_transforms()

        # load dataset configuration
        self.conf = config
        self.batch_size = batch_size
        self.root_dir = self.conf["root_dir"]
        self.cache_read = self.conf.get("cache_read", False)
        self.multi_lights = self.conf.get("multi_lights", False)
        self.augment_light_map = self.conf.get("augment_light_map", False)
        self.with_door_window = self.conf.get("with_door_window", False)

        self.scale_factor = self.conf["scale_factor"]
        self.near = self.conf["near"]
        self.far = self.conf["far"]

        # use scene center to normalize poses
        self.pose_avg = np.concatenate(
            [np.eye(3), np.array(self.conf["scene_center"])[:, None]], 1
        )

        # remove black border caused by image undistortion
        border = self.conf["border_mask"]
        w, h = self.img_wh
        self.img_size = w * h
        bmask = np.ones((h, w))
        bmask[:border, :] = 0
        bmask[-border:, :] = 0
        bmask[:, :border] = 0
        bmask[:, -border:] = 0
        self.bmask = self.transform(bmask).bool()

        # self.white_back = False
        # self.white_back = split != 'train'
        self.white_back = True

        if self.multi_lights:
            # get lights
            light_env_folders = list_dir(
                os.path.join(self.root_dir, "background_multi_lights")
            )
            light_env_ids = range(len(light_env_folders))
            if split == "val":  # only retain specific light for debugging
                validate_light_inds = self.conf["validate_lights"]
                light_env_folders = [light_env_folders[i] for i in validate_light_inds]
                light_env_ids = [light_env_ids[i] for i in validate_light_inds]

        # augment with diffuse map
        if self.augment_light_map:
            # get diffuse map files
            self.hdr_gallary_diffuse_map_dir = self.conf["hdr_gallary_diffuse_map_dir"]
            diffuse_maps = list_dir(self.hdr_gallary_diffuse_map_dir)
            diffuse_maps = diffuse_maps[:: self.conf["diffuse_map_skip"]]
            # horizontal offset
            num_augment_offset = self.conf["N_diffuse_map_augment_offset"]
            augment_offsets = np.linspace(0.0, 1.0, num_augment_offset).tolist()
            # construct pairs
            self.diffuse_map_augment_pairs = [
                (d_map, h_offset)
                for d_map, h_offset in itertools.product(diffuse_maps, augment_offsets)
            ]
            self.diffuse_map_augment_pairs.insert(
                0, (None, None)
            )  # first light_env is original
            light_env_ids = range(len(self.diffuse_map_augment_pairs))
            print("Total num of light_envs =", len(light_env_ids))
            if split == "val":
                validate_light_inds = self.conf["validate_lights"]
                light_env_ids = [
                    light_env_ids[i * num_augment_offset] for i in validate_light_inds
                ]

        self.frame_data = []
        if self.multi_lights:
            for i_light, (light_env_id, light_env_name) in enumerate(
                zip(light_env_ids, light_env_folders)
            ):
                full_obj_path = os.path.join(
                    self.root_dir,
                    "background_multi_lights",
                    light_env_name,
                )
                self.read_meta_background(full_obj_path, 0, light_env_id)
                remove_stdout_line(1)
        elif self.augment_light_map:
            for i_light, light_env_id in enumerate(light_env_ids):
                full_obj_path = os.path.join(
                    self.root_dir,
                    "background_with_door_window"
                    if self.with_door_window
                    else "background",
                )
                self.read_meta_background(full_obj_path, 0, light_env_id)
                remove_stdout_line(1)
        else:
            full_obj_path = os.path.join(
                self.root_dir,
                "background_with_door_window"
                if self.with_door_window
                else "background",
            )
            self.read_meta_background(full_obj_path, 0)
            remove_stdout_line(1)

    def read_meta_background(self, base_dir, bg_id, light_env_id=0):
        # Step 1. generate rays for each image in camera coordinate
        w, h = self.img_wh
        # ray directions for all pixels
        if not hasattr(self, "directions"):
            self.directions = get_ray_directions_equirectangular(h, w)  # (h, w, 3)
            self.direction_orig_norm = torch.norm(self.directions, dim=-1, keepdim=True)

        # Step 2. filter image list via preset parameters and observation check
        validate_idx = self.conf["validate_idx"]
        total_num_frames = self.conf["total_num_frames"]
        train_skip_step = self.conf["train_skip_step"]

        if self.split == "train":
            self.frame_indices = range(0, total_num_frames, train_skip_step)
            self.frame_indices = list(
                filter(lambda x: (x != validate_idx), self.frame_indices)
            )
        elif self.split == "val":
            self.frame_indices = [validate_idx]

        # Step 4. create buffer of all rays and rgb data
        instance_ids = [bg_id]

        for frame_idx in self.frame_indices:
            for i_inst, instance_id in enumerate(instance_ids):
                print(
                    "\rRead meta {:05d} : {:05d}".format(
                        frame_idx, len(self.frame_indices)
                    ),
                    end="",
                )
                frame_base_dir = os.path.join(base_dir, f"{frame_idx:05d}")
                if self.cache_read:  # cache frame reading
                    sample = self.read_frame_data_background(
                        frame_base_dir, frame_idx, instance_id, light_env_id, 0
                    )
                    self.frame_data.append(sample)
                    assert False, "cache_read does not support frame_ids"
                else:
                    self.frame_data.append(
                        {
                            "frame_base_dir": frame_base_dir,
                            "frame_idx": frame_idx,
                            "instance_id": instance_id,
                            "light_env_id": light_env_id,
                            "frame_id": frame_idx,
                        }
                    )
        print("")

    def define_transforms(self):
        self.transform = T.ToTensor()

    def read_frame_data_background(
        self, base_dir, frame_idx, instance_id, light_env_id, frame_id
    ):
        frame = read_json(os.path.join(base_dir, "data.json"))

        valid_mask = self.bmask.flatten()  # (h*w) valid_mask

        instance_mask = torch.ones_like(valid_mask).bool()
        instance_mask_weight = torch.ones_like(valid_mask)

        # Original poses has rotation in form "right down forward", change to NDC "right up back"
        fix_rot = np.array([1, 0, 0, 0, -1, 0, 0, 0, -1]).reshape(3, 3)
        pose = np.array(frame["camera"]["cam3d2world"]).reshape(4, 4)
        pose[:3, :3] = pose[:3, :3] @ fix_rot

        # centralize and rescale
        pose = center_pose_from_avg(self.pose_avg, pose)
        pose[:, 3] /= self.scale_factor

        c2w = torch.FloatTensor(pose)[:3, :4]

        img_path = os.path.join(base_dir, "rgb.png")
        if not os.path.exists(img_path):
            print("Skip file which does not exist", img_path)
            return None

        # read image
        img = Image.open(img_path)
        img = img.resize(self.img_wh, Image.LANCZOS)
        img = np.array(img)

        # read depth
        depth = cv2.imread(os.path.join(base_dir, "depth.png"), cv2.IMREAD_ANYDEPTH)
        # specify for ig background
        depth = (depth >> 3) | (depth << 13)
        depth = depth.astype(np.float32) * 1e-3
        if depth is None:
            depth = np.zeros((self.img_wh[1], self.img_wh[0]))
        else:
            depth = cv2.resize(depth, self.img_wh, interpolation=cv2.INTER_NEAREST)
            # depth[depth>4] = 0

        # apply light map augmentation
        if self.augment_light_map and light_env_id != 0:
            map_name, h_offset = self.diffuse_map_augment_pairs[light_env_id]
            # read normal
            normal_path = os.path.join(base_dir, "normal.png")
            normal_map = imageio.imread(normal_path)
            normal_map = cv2.resize(
                normal_map, self.img_wh, interpolation=cv2.INTER_NEAREST
            )
            # read light
            light_map = cv2.imread(
                os.path.join(self.hdr_gallary_diffuse_map_dir, map_name), -1
            )
            light_map = cv2.cvtColor(light_map, cv2.COLOR_BGR2RGB)
            img = apply_light_map_augmentation(
                img=img,
                pose=pose,
                light_map=light_map,
                normal_map=normal_map,
                is_panorama=True,
                shrink_range=self.conf["diffuse_map_shrink_range"],
                horizontal_offset=h_offset,
            )

        img = self.transform(img)  # (3, H, W)
        # valid_mask = (img[-1]>0).flatten() # (H*W) valid color area
        img = img.view(3, -1).permute(1, 0)  # (H*W, 3) RGB
        # img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB
        depth = self.transform(depth).float().squeeze()  # (H, W)
        depth = depth.view(-1)  # (H*W)
        depth /= self.scale_factor
        # TODO(ybbbbt): hotfix from depth(z value) to ray step(used in nerf rendering)
        depth *= self.direction_orig_norm.view(-1)

        rays_o, rays_d = get_rays(self.directions, c2w)

        # debug depth
        if self.conf.use_statistic_near_far:
            statistic_near = np.percentile(np.unique(depth), 0.01)
            statistic_far = np.percentile(depth, 99.99)
            # print(depth, np.percentile(np.unique(depth), 0.1), np.percentile(depth, 99.9))
            # near = min(self.near, statistic_near)
            # far = max(self.far, statistic_far)
            near = max(0, statistic_near - 1)
            far = statistic_far + 1
        else:
            near = self.near
            far = self.far

        batch_near = near / self.scale_factor * torch.ones_like(rays_o[:, :1])
        batch_far = far / self.scale_factor * torch.ones_like(rays_o[:, :1])

        rays = torch.cat([rays_o, rays_d, batch_near, batch_far], 1)  # (H*W, 8)

        return {
            "rays": rays,
            "rgbs": img,
            "depths": depth,
            "c2w": c2w,
            "valid_mask": valid_mask,
            "instance_mask": instance_mask,
            "instance_mask_weight": instance_mask_weight,
            "instance_ids": torch.ones_like(depth).long() * instance_id,
            "light_env_ids": torch.ones_like(depth).long() * light_env_id,
            "frame_ids": torch.ones_like(depth).long() * frame_id,
        }

    def sample_rays_from_single_frame(self, idx):
        frame_idx = idx % len(self.frame_data)
        if self.cache_read:
            sample = copy.deepcopy(self.frame_data[frame_idx])
        else:
            frame_param = self.frame_data[frame_idx]
            sample = self.read_frame_data_background(
                frame_param["frame_base_dir"],
                frame_param["frame_idx"],
                frame_param["instance_id"],
                frame_param["light_env_id"],
                frame_param["frame_id"],
            )

        sample["frame_idx"] = frame_idx

        need_backup = False
        # backup origional rgb and depth for warping
        if need_backup:
            sample["rgbs_orig"] = sample["rgbs"]
            sample["rays_full"] = sample["rays"]
            sample["image_wh"] = self.img_wh
            sample["depths_orig"] = sample["depths"]
            sample["valid_mask_orig"] = sample["valid_mask"]
            sample["instance_mask_orig"] = sample["instance_mask"]
            sample["instance_mask_weight_orig"] = sample["instance_mask_weight"]
            sample["instance_ids_orig"] = sample["instance_ids"]
            sample["light_env_ids_orig"] = sample["light_env_ids"]
            sample["frame_ids_orig"] = sample["frame_ids"]

        if self.split == "train":
            # random select rays
            N_full_rays = self.img_wh[0] * self.img_wh[1]
            # importance sampling on the object area
            select_inds = np.random.choice(
                N_full_rays, size=(self.batch_size,), replace=False
            )
        else:
            select_inds = np.arange(self.img_wh[0] * self.img_wh[1])

        sample["rays"] = sample["rays"][select_inds]
        sample["rgbs"] = sample["rgbs"][select_inds]
        sample["depths"] = sample["depths"][select_inds]
        sample["valid_mask"] = sample["valid_mask"][select_inds]
        sample["instance_mask"] = sample["instance_mask"][select_inds]
        sample["instance_mask_weight"] = sample["instance_mask_weight"][select_inds]
        sample["instance_ids"] = sample["instance_ids"][select_inds]
        sample["light_env_ids"] = sample["light_env_ids"][select_inds]
        sample["frame_ids"] = sample["frame_ids"][select_inds]
        sample["select_inds"] = select_inds

        return sample

    def __len__(self):
        if self.split == "train":
            return len(self.frame_data) * self.img_size // self.batch_size
        elif self.split == "val":
            # return 8 # only validate 8 images (to support <=8 gpus)
            return len(self.frame_data)

    def __getitem__(self, idx):
        if self.split == "train":  # use data in the buffers
            return self.sample_rays_from_single_frame(idx)
        else:  # create data for each image separately
            sample = self.sample_rays_from_single_frame(idx)

        return sample


if __name__ == "__main__":
    from omegaconf import OmegaConf

    conf = OmegaConf.load("config/igibson_Beechwood_1_int_bedroom_0_bg.yml")
    dataset = IGBackground_Single(
        batch_size=1024, split="train", config=conf["dataset"]
    )
    item = dataset[0]
    # import ipdb; ipdb.set_trace()
