import torch
import numpy as np
import cv2
from models import perceptual_model
from models.perceptual_model import get_perceptual_loss, VGG16_for_Perceptual
from typing import List, Optional, Any, Dict, Union

# import lpips

# loss_fn_vgg = lpips.LPIPS(net="vgg").cuda()


def get_mask_bbox(mask):
    # crop image
    true_indices = np.nonzero(mask)
    min_h, min_w = np.min(true_indices[0]), np.min(true_indices[1])
    max_h, max_w = np.max(true_indices[0]), np.max(true_indices[1])
    # print(min_h, min_w)
    # print(max_h, max_w)
    # img = img[min_h:max_h+1,min_w:max_w+1,:]
    return min_h, max_h, min_w, max_w


def patch_perceptual_loss(
    perceptual_net: VGG16_for_Perceptual,
    input_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
    all_obj_info: dict(),
    instance_mask: torch.Tensor,
    img_wh,
):
    w, h = img_wh
    perceptual_loss_dict = {}
    for idx, obj_info in all_obj_info.items():
        obj_id = obj_info["obj_id"]
        inst_mask_obj = instance_mask == obj_id

        kernel = np.ones((7, 7), np.uint8)
        inst_mask_obj = cv2.dilate(
            inst_mask_obj.reshape(h, w).astype(np.uint8), kernel, iterations=1
        ).astype(bool)
        min_h, max_h, min_w, max_w = get_mask_bbox(inst_mask_obj.reshape(h, w))

        full_ind = np.arange(w * h).reshape(h, w)
        select_ind = full_ind[min_h : max_h + 1, min_w : max_w + 1].reshape(-1)
        crop_size = (max_h - min_h + 1, max_w - min_w + 1)

        # crop according to mask
        crop_h, crop_w = crop_size

        input_patch = input_rgb[select_ind].permute(1, 0).view(1, 3, crop_h, crop_w)
        target_patch = (
            target_rgb[select_ind].permute(1, 0).view(1, 3, crop_h, crop_w).cuda()
        )

        # loss = get_perceptual_loss(
        #     perceptual_net,
        #     input_patch,
        #     target_patch,
        #     # low_level=False,
        #     low_level=True,
        # )

        # loss = loss_fn_vgg((input_patch - 0.5) * 2, (target_patch - 0.5) * 2)

        # perceptual_loss_dict[f"perceptual_patch_loss_{obj_id}"] = loss
        # print(loss)
        # cv2.imwrite(
        #     f"debug/input_rgb_{obj_id}.png",
        #     (input_rgb[select_ind] * 255)
        #     .view(crop_h, crop_w, 3)
        #     .detach()
        #     .cpu()
        #     .numpy()
        #     .astype(np.uint8),
        # )
        # cv2.imwrite(
        #     f"debug/target_rgb{obj_id}.png",
        #     (target_rgb[select_ind] * 255)
        #     .view(crop_h, crop_w, 3)
        #     .detach()
        #     .cpu()
        #     .numpy()
        #     .astype(np.uint8),
        # )
    # print(perceptual_loss_dict)
    # loss = get_perceptual_loss(
    #     perceptual_net,
    #     input_rgb.permute(1, 0).view(1, 3, h, w),
    #     target_rgb.permute(1, 0).view(1, 3, h, w).cuda(),
    # )

    # cv2.imwrite(
    #     f"debug/input_rgb.png",
    #     (input_rgb * 255).view(h, w, 3).detach().cpu().numpy().astype(np.uint8),
    # )
    # cv2.imwrite(
    #     f"debug/target_rgb.png",
    #     (target_rgb * 255).view(h, w, 3).detach().cpu().numpy().astype(np.uint8),
    # )

    # perceptual_loss_dict[f"perceptual_loss"] = loss
    return perceptual_loss_dict
