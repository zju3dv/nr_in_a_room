import sys
from kornia.geometry.transform.affwarp import scale

sys.path.append(".")

import torch
from torch.utils import data
from torch.utils.data import Dataset
import json
import numpy as np
import os
import cv2
import imageio
import itertools
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


class IGObject_Single(Dataset):
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

        self.scale_factor = self.conf["scale_factor"]
        self.scale_factor_dict = self.conf.get("scale_factor_dict", {})
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

        self.training_obj_ids = self.conf.training_obj_ids
        self.val_obj_ids = self.conf.val_obj_ids

        if self.multi_lights:
            # get lights
            light_env_folders = list_dir(
                os.path.join(self.root_dir, "objects_multi_lights")
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
            sys.stdout.write("\033[K")
            print("Total num of light_envs =", len(light_env_ids))
            if split == "val":
                validate_light_inds = self.conf["validate_lights"]
                light_env_ids = [
                    light_env_ids[i * num_augment_offset] for i in validate_light_inds
                ]

        self.frame_data = []
        scene_info = read_json(
            os.path.join(self.root_dir, self.conf.scene_info_path, "data.json")
        )
        for obj in tqdm(scene_info["objs"]):
            obj_path = obj["model_path"]
            obj_id = obj["id"]
            if split == "train" and obj_id not in self.training_obj_ids:
                continue
            if split == "val" and obj_id not in self.val_obj_ids:
                continue
            print("Read model {} : {}".format(obj_id, obj_path))
            if self.multi_lights:
                for i_light, (light_env_id, light_env_name) in enumerate(
                    zip(light_env_ids, light_env_folders)
                ):
                    full_obj_path = os.path.join(
                        self.root_dir,
                        "objects_multi_lights",
                        light_env_name,
                        "{:03d}".format(obj_id),
                        obj_path,
                    )
                    self.read_meta(full_obj_path, obj_id, light_env_id)
                    remove_stdout_line(2)
            elif self.augment_light_map:
                for i_light, light_env_id in enumerate(light_env_ids):
                    full_obj_path = os.path.join(
                        self.root_dir, "objects", "{:03d}".format(obj_id), obj_path
                    )
                    self.read_meta(full_obj_path, obj_id, light_env_id)
                    remove_stdout_line(2)
            else:
                full_obj_path = os.path.join(
                    self.root_dir, "objects", "{:03d}".format(obj_id), obj_path
                )
                self.read_meta(full_obj_path, obj_id)
                remove_stdout_line(2)

    def get_instance_mask(self, instance_path, instance_id):
        instance = cv2.resize(
            cv2.imread(instance_path, cv2.IMREAD_ANYDEPTH),
            self.img_wh,
            interpolation=cv2.INTER_NEAREST,
        )
        mask = instance == instance_id
        return mask

    def read_meta(self, base_dir, obj_id, light_env_id=0):
        # Step 0. read json files for training and testing list
        data_json_path = os.path.join(base_dir, "transforms_full.json")

        with open(data_json_path, "r") as f:
            self.meta = json.load(f)

        # Step 1. generate rays for each image in camera coordinate
        w, h = self.img_wh
        self.focal = (
            0.5 * w / np.tan(0.5 * self.meta["camera_angle_x"])
        )  # original focal length
        # when W=800

        self.focal *= (
            self.img_wh[0] / w
        )  # modify focal length to match size self.img_wh

        # ray directions for all pixels, same for all images (same H, W, focal)
        if not hasattr(self, "directions"):
            self.directions = get_ray_directions(h, w, self.focal)  # (h, w, 3)
            self.direction_orig_norm = torch.norm(self.directions, dim=-1, keepdim=True)

        # Step 2. filter image list via preset parameters and observation check
        validate_idx = self.conf["validate_idx"]
        train_skip_step = self.conf["train_skip_step"]

        if self.split == "train":
            if self.conf.get("train_split_from_all", False):  # only remove validate idx
                split_inds = np.arange(len(self.meta["frames"]))
                split_inds = split_inds[split_inds != validate_idx]
            else:  # only retain train split
                split_inds = np.loadtxt(
                    os.path.join(self.conf["split"], "train.txt")
                ).tolist()
            print("Training split count", len(split_inds))
            self.meta["frames"] = list(
                filter(lambda x: (x["idx"] in split_inds), self.meta["frames"])
            )
            # skip frames
            self.meta["frames"] = [
                self.meta["frames"][i]
                for i in np.arange(0, len(self.meta["frames"]), train_skip_step)
            ]

            frames = self.meta["frames"]
            # print(
            #     "Train idx: {} -> {}, skip: {}".format(
            #         frames[0]["idx"], frames[-1]["idx"], train_skip_step
            #     )
            # )

        elif self.split == "val":
            # we only set one frame for valid
            self.meta["frames"] = list(
                filter(lambda x: (x["idx"] == validate_idx), self.meta["frames"])
            )
            print("Valid idx: {}".format(validate_idx))

        # Step 4. create buffer of all rays and rgb data
        instance_ids = [obj_id]

        for idx, frame in enumerate(self.meta["frames"]):
            for i_inst, instance_id in enumerate(instance_ids):
                print(
                    "\rRead meta {:05d} : {:05d}, instance {:d}".format(
                        idx, len(self.meta["frames"]) - 1, instance_id
                    ),
                    end="",
                )
                if self.cache_read:  # cache frame reading
                    sample = self.read_frame_data(
                        base_dir, frame, instance_id, light_env_id, 0
                    )
                    self.frame_data.append(sample)
                    assert False, "cache_read does not support frame_ids"
                else:
                    self.frame_data.append(
                        {
                            "base_dir": base_dir,
                            "frame": frame,
                            "instance_id": instance_id,
                            "light_env_id": light_env_id,
                            "frame_id": idx,
                        }
                    )
        print("")

    def define_transforms(self):
        self.transform = T.ToTensor()

    def read_frame_data(self, base_dir, frame, instance_id, light_env_id, frame_id):

        valid_mask = self.bmask.flatten()  # (h*w) valid_mask

        # read instance mask
        if self.conf["use_instance_mask"] and instance_id != 0:
            instance_path = os.path.join(
                base_dir, f"{frame['file_path']}.{self.conf['inst_seg_tag']}.png"
            )
            instance_mask = self.get_instance_mask(instance_path, instance_id)
            # apply erode to mask
            if self.conf.get("mask_erode", 0) > 0:
                erode_kernel = self.conf["mask_erode"]
                kernel = np.ones((erode_kernel, erode_kernel), np.float32)
                instance_mask = cv2.erode(
                    instance_mask.astype(np.float32), kernel, iterations=1
                ).astype(bool)
                # ).astype(np.float32)
                # cv2.normalize(instance_mask, instance_mask, 0, 1, cv2.NORM_MINMAX)
                # cv2.imshow("Erode Image", instance_mask)
                # cv2.waitKey(0)

            if self.conf.get("soften_mask_border", False):
                instance_mask_weight = compute_distance_transfrom_weights(
                    instance_mask,
                    self.conf["soften_mask_pixel"],
                    fg_bg_balance_weight=False,
                )
            else:
                instance_mask_weight = rebalance_mask(
                    instance_mask,
                    fg_weight=self.conf["mask_fg_weight"],
                    bg_weight=self.conf["mask_bg_weight"],
                )
            instance_mask, instance_mask_weight = self.transform(instance_mask).view(
                -1
            ), self.transform(instance_mask_weight).view(-1)
        else:
            instance_mask = torch.ones_like(valid_mask).bool()
            instance_mask_weight = torch.ones_like(valid_mask)

        # Original poses has rotation in form "right down forward", change to NDC "right up back"
        # fix_rot = np.array([1, 0, 0,
        #                     0, -1, 0,
        #                     0, 0, -1]).reshape(3, 3)
        pose = np.array(frame["transform_matrix"])
        # pose[:3, :3] = pose[:3, :3] @ fix_rot

        # centralize and rescale
        pose = center_pose_from_avg(self.pose_avg, pose)
        scale_factor = self.scale_factor_dict.get(str(instance_id), self.scale_factor)
        pose[:, 3] /= scale_factor

        c2w = torch.FloatTensor(pose)[:3, :4]

        # read image
        img_path = os.path.join(base_dir, f"{frame['file_path']}.png")
        if not os.path.exists(img_path):
            print("Skip file which does not exist", img_path)
            return None
        img = Image.open(img_path)
        img = img.resize(self.img_wh, Image.LANCZOS)
        img = np.array(img)

        # read depth
        depth = cv2.imread(
            os.path.join(base_dir, f"{frame['file_path']}.depth.png"),
            cv2.IMREAD_ANYDEPTH,
        )
        depth = depth * 1e-3
        if depth is None:
            depth = np.zeros((self.img_wh[1], self.img_wh[0]))
        else:
            depth = cv2.resize(depth, self.img_wh, interpolation=cv2.INTER_NEAREST)

        # apply light map augmentation
        if self.augment_light_map and light_env_id != 0:
            map_name, h_offset = self.diffuse_map_augment_pairs[light_env_id]
            # read normal
            normal_path = os.path.join(base_dir, f"{frame['file_path']}.normal.png")
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
                is_panorama=False,
                shrink_range=self.conf["diffuse_map_shrink_range"],
                horizontal_offset=h_offset,
            )
        img = self.transform(img)  # (3, H, W)
        # valid_mask = (img[-1]>0).flatten() # (H*W) valid color area
        img = img.view(3, -1).permute(1, 0)  # (H*W, 3) RGB
        # img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB

        # depth[depth>4] = 0
        depth = self.transform(depth).float().squeeze()  # (H, W)
        depth = depth.view(-1)  # (H*W)
        scale_factor = self.scale_factor_dict.get(str(instance_id), self.scale_factor)
        depth /= scale_factor
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

        batch_near = near / scale_factor * torch.ones_like(rays_o[:, :1])
        batch_far = far / scale_factor * torch.ones_like(rays_o[:, :1])

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
            "file_path": frame["file_path"],
        }

    def get_mask_bbox(self, mask):
        # crop image
        true_indices = np.nonzero(mask)
        min_h, min_w = np.min(true_indices[0]), np.min(true_indices[1])
        max_h, max_w = np.max(true_indices[0]), np.max(true_indices[1])
        # print(min_h, min_w)
        # print(max_h, max_w)
        # img = img[min_h:max_h+1,min_w:max_w+1,:]
        return min_h, max_h, min_w, max_w

    def sample_rays_from_single_frame(self, idx):
        frame_idx = idx % len(self.frame_data)
        if self.cache_read:
            sample = copy.deepcopy(self.frame_data[frame_idx])
        else:
            frame_param = self.frame_data[frame_idx]
            sample = self.read_frame_data(
                frame_param["base_dir"],
                frame_param["frame"],
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

        object_sampling_ratio = self.conf.object_sampling_ratio

        if self.split == "train":
            # random select rays
            N_full_rays = self.img_wh[0] * self.img_wh[1]
            # importance sampling on the object area
            if object_sampling_ratio > 0:
                in_obj_inds = np.arange(N_full_rays)[sample["instance_mask"]]
                out_obj_inds = np.arange(N_full_rays)[~sample["instance_mask"]]
                # handle when in-object area is smaller than batch size
                N_in_obj_sample = min(
                    int(object_sampling_ratio * self.batch_size), in_obj_inds.shape[0]
                )
                select_inds = np.random.choice(
                    in_obj_inds, size=(N_in_obj_sample,), replace=False
                )
                # concatenate in and out sampling
                if N_in_obj_sample < self.batch_size:
                    select_out_inds = np.random.choice(
                        out_obj_inds,
                        size=(self.batch_size - N_in_obj_sample),
                        replace=False,
                    )
                    select_inds = np.concatenate([select_inds, select_out_inds])
            else:
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
            # return len(self.val_obj_ids)
            return len(self.frame_data)
        # return len(self.meta['frames'])

    def __getitem__(self, idx):
        if self.split == "train":  # use data in the buffers
            return self.sample_rays_from_single_frame(idx)
        else:  # create data for each image separately
            sample = self.sample_rays_from_single_frame(idx)

        return sample


if __name__ == "__main__":
    from omegaconf import OmegaConf

    conf = OmegaConf.load("config/igibson_Beechwood_1_int_bedroom_0.yml")
    dataset = IGObject_Single(batch_size=1024, split="train", config=conf["dataset"])
    item = dataset[0]
    # import ipdb; ipdb.set_trace()
