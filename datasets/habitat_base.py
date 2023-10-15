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
from PIL import Image
from torchvision import transforms as T

from datasets.ray_utils import *
from datasets.geo_utils import *
from datasets.image_utils import *
from utils.util import read_yaml, write_idx


class HabitatBase(Dataset):
    def __init__(self, split="train", img_wh=(640, 480), config=None):
        self.split = split
        # assert img_wh[0] == img_wh[1], 'image width must equal image height!'
        self.img_wh = img_wh
        self.define_transforms()

        # load dataset configuration
        self.conf = config
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
        bmask = np.ones((h, w))
        bmask[:border, :] = 0
        bmask[-border:, :] = 0
        bmask[:, :border] = 0
        bmask[:, -border:] = 0
        self.bmask = self.transform(bmask).bool()

        # self.white_back = False
        self.white_back = split != "train"

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
        if self.split == "train":
            self.image_paths = []
            self.all_rays = []
            self.all_rgbs = []
            self.all_depths = []
            self.all_valid_masks = []
            self.all_instance_masks = []
            self.all_instance_masks_weight = []
            self.all_frame_indices = []
            self.all_instance_ids = []

            self.instance_ids = self.conf["instance_id"]
            # instance_ids.append(0) # for the completeness of background images
            image_idx_set = set()

            for idx, frame in enumerate(self.meta["frames"]):
                curr_frame_instance_masks = []
                curr_frame_instance_masks_weight = []
                curr_frame_instance_ids = []
                for i_inst, instance_id in enumerate(self.instance_ids):
                    print(
                        "\rRead meta {:05d} : {:05d} instance {:01d}".format(
                            idx, len(self.meta["frames"]) - 1, instance_id
                        ),
                        end="",
                    )
                    sample = self.read_frame_data(
                        frame, instance_id, read_instance_only=(i_inst != 0)
                    )
                    # skip duplicates
                    if instance_id == 0 and idx in image_idx_set:
                        print("continue for bgs")
                        continue
                    if sample is None:
                        print(
                            "\nSkip frame {} with instance {}".format(idx, instance_id)
                        )
                        continue
                    image_idx_set.add(idx)
                    # only save for the first
                    if i_inst == 0:
                        self.all_rays += [sample["rays"]]
                        self.all_rgbs += [sample["rgbs"]]
                        self.all_depths += [sample["depths"]]
                        self.all_valid_masks += [sample["valid_mask"]]
                        self.all_frame_indices += [
                            torch.ones_like(sample["valid_mask"]) * idx
                        ]

                    curr_frame_instance_masks += [sample["instance_mask"]]
                    curr_frame_instance_masks_weight += [sample["instance_mask_weight"]]
                    curr_frame_instance_ids += [sample["instance_ids"]]

                self.all_instance_masks += [torch.stack(curr_frame_instance_masks, -1)]
                self.all_instance_masks_weight += [
                    torch.stack(curr_frame_instance_masks_weight, -1)
                ]
                self.all_instance_ids += [torch.stack(curr_frame_instance_ids, -1)]

            print("")
            self.all_rays = torch.cat(
                self.all_rays, 0
            )  # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(
                self.all_rgbs, 0
            )  # (len(self.meta['frames])*h*w, 3)
            self.all_depths = torch.cat(
                self.all_depths, 0
            )  # (len(self.meta['frames])*h*w)
            self.all_valid_masks = torch.cat(
                self.all_valid_masks, 0
            )  # (len(self.meta['frames])*h*w)
            self.all_instance_masks = torch.cat(
                self.all_instance_masks, 0
            )  # (len(self.meta['frames])*h*w)
            self.all_instance_masks_weight = torch.cat(
                self.all_instance_masks_weight, 0
            )  # (len(self.meta['frames])*h*w)
            self.all_frame_indices = torch.cat(
                self.all_frame_indices, 0
            ).long()  # (len(self.meta['frames])*h*w)
            self.all_instance_ids = torch.cat(
                self.all_instance_ids, 0
            ).long()  # (len(self.meta['frames])*h*w)

    def define_transforms(self):
        self.transform = T.ToTensor()

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

    def __len__(self):
        if self.split == "train":
            return len(self.all_rays)
        if self.split == "val":
            # return 8 # only validate 8 images (to support <=8 gpus)
            return 1
        return len(self.meta["frames"])

    def __getitem__(self, idx):
        if self.split == "train":  # use data in the buffers
            rand_instance_id = torch.randint(0, len(self.instance_ids), (1,))
            sample = {
                "rays": self.all_rays[idx],
                "rgbs": self.all_rgbs[idx],
                "depths": self.all_depths[idx],
                "valid_mask": self.all_valid_masks[idx],
                "instance_mask": self.all_instance_masks[idx, rand_instance_id],
                "instance_mask_weight": self.all_instance_masks_weight[
                    idx, rand_instance_id
                ],
                "frame_idx": self.all_frame_indices[idx],
                "instance_ids": self.all_instance_ids[idx, rand_instance_id],
            }

        else:  # create data for each image separately
            # frame = self.meta['frames'][idx]
            frame = self.meta["frames"][0]
            sample = self.read_frame_data(frame, self.conf["val_instance_id"])

            assert (
                not sample is None
            ), "val image does not have enough areas for val_instance_id"
            # TODO(ybbbbt): pseudo frame index for appearance encoding
            sample["frame_idx"] = torch.ones_like(sample["depths"]).long()

        return sample
