import sys

sys.path.append(".")

from numpy.core.defchararray import center
from numpy.core.shape_base import stack
import torch
from torch.utils import data
from torch.utils.data import Dataset
import json
import numpy as np
import os
import cv2
import random
import copy
from collections import defaultdict
from PIL import Image
from torchvision import transforms as T

from datasets.ray_utils import *
from datasets.geo_utils import *
from datasets.image_utils import *
from utils.util import read_yaml, write_idx
from models.perceptual_model import CLIP_for_Perceptual


class HabitatSingle(Dataset):
    def __init__(self, batch_size, split="train", img_wh=(640, 480), config=None):
        self.split = split
        # assert img_wh[0] == img_wh[1], 'image width must equal image height!'
        self.img_wh = img_wh
        self.define_transforms()

        # load dataset configuration
        self.conf = config
        self.batch_size = batch_size
        self.root_dir = self.conf["root_dir"]
        # self.scene_id = self.conf['scene_id']
        # if split == 'train':
        #     print('-' * 40)
        #     print(json.dumps(self.conf, sort_keys=True, indent=2))

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
        self.white_back = split != "train"

        self.perceptual_net = CLIP_for_Perceptual()

        self.read_meta()

    def get_instance_mask(self, instance_path, instance_id):
        instance = cv2.resize(
            cv2.imread(instance_path, cv2.IMREAD_ANYDEPTH),
            self.img_wh,
            interpolation=cv2.INTER_NEAREST,
        )
        mask = instance == instance_id
        return mask

    def read_meta(self):
        # Step 0. read json files for training and testing list
        data_json_path = os.path.join(self.root_dir, "transforms_full.json")
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
        self.directions = get_ray_directions(h, w, self.focal)  # (h, w, 3)

        self.direction_orig_norm = torch.norm(self.directions, dim=-1, keepdim=True)

        # Step 2. filter image list via preset parameters and observation check
        validate_idx = self.conf["validate_idx"]
        train_skip_step = self.conf["train_skip_step"]

        if self.split == "train":
            # only retain train split
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
            print(
                "Train idx: {} -> {}, skip: {}".format(
                    frames[0]["idx"], frames[-1]["idx"], train_skip_step
                )
            )

        elif self.split == "val":
            # we only set one frame for valid
            self.meta["frames"] = list(
                filter(lambda x: (x["idx"] == validate_idx), self.meta["frames"])
            )
            print("Valid idx: {}".format(validate_idx))

        # Step 4. create buffer of all rays and rgb data
        # if self.split == 'train':
        if True:
            self.instance_ids = self.conf["instance_id"]
            if self.split == "val":
                self.instance_ids = [self.conf["val_instance_id"]]
            self.frame_data = []
            self.instance_embedding = defaultdict(list)

            for idx, frame in enumerate(self.meta["frames"]):
                for i_inst, instance_id in enumerate(self.instance_ids):
                    print(
                        "\rRead meta {:05d} : {:05d}, instance {:d}".format(
                            idx, len(self.meta["frames"]) - 1, instance_id
                        ),
                        end="",
                    )
                    sample = self.read_frame_data(
                        frame, instance_id, read_instance_only=(i_inst != 0)
                    )
                    # precompute instance patch embedding
                    self.instance_embedding[str(instance_id)].append(
                        self.compute_instance_embedding(sample)
                    )
                    self.frame_data.append(sample)
            print("")

    def define_transforms(self):
        self.transform = T.ToTensor()

    def compute_instance_embedding(self, frame):
        w, h = self.img_wh
        rgb = frame["rgbs"].clone()
        # mask others to be white
        rgb[~frame["instance_mask"]] = 1
        # crop to fit each instance
        min_h, max_h, min_w, max_w = self.get_mask_bbox(
            frame["instance_mask"].view(h, w).numpy()
        )

        full_ind = np.arange(h * w).reshape(h, w)
        instance_ind = full_ind[min_h : max_h + 1, min_w : max_w + 1].reshape(-1)
        crop_size = (max_h - min_h + 1, max_w - min_w + 1)
        crop_h, crop_w = crop_size
        rgb = rgb[instance_ind].permute(1, 0).view(1, 3, crop_h, crop_w).cuda()
        # emb: [512]
        with torch.no_grad():
            emb = self.perceptual_net.compute_img_embedding(rgb).squeeze(0).cpu()
        return emb

    def read_frame_data(self, frame, instance_id, read_instance_only=False):

        valid_mask = self.bmask.flatten()  # (h*w) valid_mask

        # read instance mask
        if self.conf["use_instance_mask"] and instance_id != 0:
            instance_path = os.path.join(
                self.root_dir, f"{frame['file_path']}.{self.conf['inst_seg_tag']}.png"
            )
            instance_mask = self.get_instance_mask(instance_path, instance_id)
            # instance_mask_weight = compute_distance_transfrom_weights(
            #     instance_mask, uncertain_pixel_distance=0.05*self.img_wh[0], fg_bg_balance_weight=True)
            # instance_mask_weight = compute_distance_transfrom_weights(
            #     instance_mask, uncertain_pixel_distance=0.05*self.img_wh[0], fg_bg_balance_weight=True, fg_weight=1.0, bg_weight=0.05)
            # instance_mask, uncertain_pixel_distance=0.05*self.img_wh[0], fg_bg_balance_weight=False)
            instance_mask_weight = rebalance_mask(
                instance_mask,
                fg_weight=self.conf["mask_fg_weight"],
                bg_weight=self.conf["mask_bg_weight"],
            )
            # instance_mask_weight = rebalance_mask(instance_mask,
            #                                       fg_weight=1.0,
            #                                       bg_weight=0.05)
            # instance_mask_weight = rebalance_mask(instance_mask)
            instance_mask, instance_mask_weight = self.transform(instance_mask).view(
                -1
            ), self.transform(instance_mask_weight).view(-1)
            # add this to disbale weighting
            # instance_mask_weight = torch.ones_like(valid_mask)
            # skip training batch for which does not contain enough instance info
            # if instance_mask.sum() < self.img_wh[0] * self.img_wh[1] * 0.2 and self.split == 'train':
            #     # return None
            #     instance_mask_weight[...] = 0
        else:
            instance_mask = torch.ones_like(valid_mask).bool()
            instance_mask_weight = torch.zeros_like(valid_mask)

        if read_instance_only:
            return {
                "instance_mask": instance_mask,
                "instance_mask_weight": instance_mask_weight,
                "instance_ids": torch.ones_like(instance_mask).long() * instance_id,
            }

        # Original poses has rotation in form "right down forward", change to NDC "right up back"
        # fix_rot = np.array([1, 0, 0,
        #                     0, -1, 0,
        #                     0, 0, -1]).reshape(3, 3)
        pose = np.array(frame["transform_matrix"])
        # pose[:3, :3] = pose[:3, :3] @ fix_rot

        # centralize and rescale
        pose = center_pose_from_avg(self.pose_avg, pose)
        pose[:, 3] /= self.scale_factor

        c2w = torch.FloatTensor(pose)[:3, :4]

        img_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
        if not os.path.exists(img_path):
            print("Skip file which does not exist", img_path)
            return None

        img = Image.open(os.path.join(self.root_dir, f"{frame['file_path']}.png"))
        img = img.resize(self.img_wh, Image.LANCZOS)
        img = self.transform(img)  # (3, H, W)
        # valid_mask = (img[-1]>0).flatten() # (H*W) valid color area
        img = img.view(3, -1).permute(1, 0)  # (H*W, 3) RGB
        # img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB

        depth = cv2.imread(
            os.path.join(self.root_dir, f"{frame['file_path']}.depth.png"),
            cv2.IMREAD_ANYDEPTH,
        )
        if depth is None:
            depth = np.zeros((self.img_wh[1], self.img_wh[0]))
        else:
            depth = (
                cv2.resize(depth, self.img_wh, interpolation=cv2.INTER_NEAREST) * 1e-3
            )
            # depth[depth>4] = 0
        depth = self.transform(depth).float().squeeze()  # (H, W)
        depth = depth.view(-1)  # (H*W)
        depth /= self.scale_factor
        # TODO(ybbbbt): hotfix from depth(z value) to ray step(used in nerf rendering)
        depth *= self.direction_orig_norm.view(-1)

        rays_o, rays_d = get_rays(self.directions, c2w)

        batch_near = self.near / self.scale_factor * torch.ones_like(rays_o[:, :1])
        batch_far = self.far / self.scale_factor * torch.ones_like(rays_o[:, :1])

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
        sample = copy.deepcopy(self.frame_data[frame_idx])

        sample["frame_idx"] = frame_idx

        # backup origional rgb and depth for warping
        sample["rgbs_orig"] = sample["rgbs"]
        sample["rays_full"] = sample["rays"]
        sample["image_wh"] = self.img_wh
        sample["depths_orig"] = sample["depths"]
        sample["valid_mask_orig"] = sample["valid_mask"]
        sample["instance_mask_orig"] = sample["instance_mask"]
        sample["instance_mask_weight_orig"] = sample["instance_mask_weight"]
        sample["instance_ids_orig"] = sample["instance_ids"]

        # random select rays
        if self.split == "train":
            select_inds = np.random.choice(
                self.img_wh[0] * self.img_wh[1], size=(self.batch_size,), replace=False
            )
        else:
            select_inds = np.arange(self.img_wh[0] * self.img_wh[1])

        sample["rays"] = sample["rays_full"][select_inds]
        sample["rgbs"] = sample["rgbs"][select_inds]
        sample["depths"] = sample["depths"][select_inds]
        sample["valid_mask"] = sample["valid_mask"][select_inds]
        sample["instance_mask"] = sample["instance_mask"][select_inds]
        sample["instance_mask_weight"] = sample["instance_mask_weight"][select_inds]
        sample["instance_ids"] = sample["instance_ids"][select_inds]
        sample["select_inds"] = select_inds

        # generate patch based idx
        w, h = self.img_wh
        min_h, max_h, min_w, max_w = self.get_mask_bbox(
            sample["instance_mask_orig"].view(h, w).numpy()
        )

        full_ind = np.arange(h * w).reshape(h, w)
        # TODO(ybbbbt): fixme, patch skip
        patch_skip = 1
        select_inds_patch = full_ind[
            min_h : max_h + 1 : patch_skip, min_w : max_w + 1 : patch_skip
        ].reshape(-1)
        crop_size = (
            (max_h - min_h + 1) // patch_skip,
            (max_w - min_w + 1) // patch_skip,
        )
        sample["rays_patch"] = sample["rays_full"][select_inds_patch]
        sample["rgbs_patch"] = sample["rgbs_orig"][select_inds_patch]
        sample["depths_patch"] = sample["depths_orig"][select_inds_patch]
        sample["valid_mask_patch"] = sample["valid_mask_orig"][select_inds_patch]
        sample["instance_mask_patch"] = sample["instance_mask_orig"][select_inds_patch]
        sample["instance_mask_weight_patch"] = sample["instance_mask_weight_orig"][
            select_inds_patch
        ]
        sample["instance_ids_patch"] = sample["instance_ids_orig"][select_inds_patch]
        sample["select_inds_patch"] = select_inds_patch
        sample["patch_hw"] = crop_size

        # we assume instance id is the same for each frame
        instance_id = int(sample["instance_ids"][0])
        # randomly select one patch embedding
        sample["ref_patch_embedding"] = random.choice(
            self.instance_embedding[str(instance_id)]
        )

        return sample

    def __len__(self):
        if self.split == "train":
            return len(self.frame_data) * self.img_size // self.batch_size
        if self.split == "val":
            # return 8 # only validate 8 images (to support <=8 gpus)
            return 1
        return len(self.meta["frames"])

    def __getitem__(self, idx):
        if self.split == "train":  # use data in the buffers
            return self.sample_rays_from_single_frame(idx)
        else:  # create data for each image separately
            # frame = self.meta['frames'][idx]
            # frame = self.meta['frames'][0]
            # sample = self.read_frame_data(frame, self.conf['val_instance_id'])

            # assert not sample is None, 'val image does not have enough areas for val_instance_id'
            # # TODO(ybbbbt): pseudo frame index for appearance encoding
            # sample['frame_idx'] = torch.ones_like(sample['depths']).long()
            sample = self.sample_rays_from_single_frame(0)

        return sample


if __name__ == "__main__":
    from omegaconf import OmegaConf

    conf = OmegaConf.load("config/habitat_castle_box.yml")
    dataset = HabitatSingle(batch_size=1024, split="train", config=conf["dataset"])
    item = dataset[0]
    # import ipdb; ipdb.set_trace()
