import torch
from kornia.losses import ssim as dssim
from argparse import ArgumentParser
import numpy as np
import os
import cv2

# import skimage
from skimage import metrics


def mse(image_pred, image_gt, valid_mask=None, reduction="mean"):
    value = (image_pred - image_gt) ** 2
    if valid_mask is not None:
        value = value[valid_mask]
    if reduction == "mean":
        return torch.mean(value)
    return value


def psnr(image_pred, image_gt, valid_mask=None, reduction="mean"):
    return -10 * torch.log10(mse(image_pred, image_gt, valid_mask, reduction))


def ssim(image_pred, image_gt, reduction="mean"):
    """
    image_pred and image_gt: (1, 3, H, W)
    """
    dssim_ = dssim(image_pred, image_gt, 3, reduction)  # dissimilarity in [0, 1]
    return 1 - 2 * dssim_  # in [-1, 1]


def compute_psnr(p, t):
    """Compute PSNR of model image predictions.
    :param prediction: Return value of forward pass.
    :param ground_truth: Ground truth.
    :return: (psnr, ssim): tuple of floats
    """
    ssim = metrics.structural_similarity(p, t, multichannel=True, data_range=1)
    psnr = metrics.peak_signal_noise_ratio(p, t, data_range=1)
    return ssim, psnr


import lpips

loss_fn_alex = lpips.LPIPS(net="alex")  # best forward scores
# loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization

# img0 = torch.zeros(1,3,64,64) # image should be RGB, IMPORTANT: normalized to [-1,1]
# img1 = torch.zeros(1,3,64,64)
# d = loss_fn_alex(img0, img1)


def get_opts():
    parser = ArgumentParser()
    # parser.add_argument("--eval_idx_file", type=str)
    parser.add_argument("--gt_img_path", type=str)
    parser.add_argument("--res_img_path", type=str)
    return parser.parse_args()


def load_image_to_tensor(path):
    if not os.path.exists(path):
        print("cannot open", path)
    img = cv2.imread(path)
    img = cv2.resize(img, dsize=(1024, 512), interpolation=cv2.INTER_AREA)
    # remove black board
    border = 10
    # w, h = 640, 480
    # w, h = img.shape
    h, w, _ = img.shape
    print(h, w)
    bmask = np.ones((h, w))
    bmask[:border, :] = 0
    bmask[-border:, :] = 0
    bmask[:, :border] = 0
    bmask[:, -border:] = 0
    # print(bmask.sum())
    img = bmask[:, :, None].astype(np.uint8) * img
    # cv2.imshow('img', img)
    # cv2.waitKey(3000)
    img = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(
        2, 0, 1
    )  # 3, h, w
    img = img / 255
    return img.float()


if __name__ == "__main__":
    args = get_opts()
    # all_psnr = []
    # all_ssim = []
    # all_lpips = []
    # for idx, frame_id in enumerate(eval_inds):
    # print(idx, frame_id)
    # print('gt')
    gt_img_path = os.path.join(args.gt_img_path)
    gt_img = load_image_to_tensor(gt_img_path)
    # print('res')
    res_img_path = os.path.join(args.res_img_path)
    res_img = load_image_to_tensor(res_img_path)

    # print(psnr(res_img, gt_img))
    ssim, psnr = compute_psnr(
        res_img.permute(1, 2, 0).numpy(), gt_img.permute(1, 2, 0).numpy()
    )
    lpips = loss_fn_alex(res_img * 2 - 1, gt_img * 2 - 1).squeeze().item()
    print(f"PSNR {psnr} SSIM {ssim} LPIPS {lpips}")
    # all_psnr.append(psnr)
    # all_ssim.append(ssim)
    # all_lpips.append(lpips)

    # N = len(eval_inds)
    # print(
    #     "Mean PSNR {} SSIM {} LPIPS {}".format(
    #         sum(all_psnr) / N, sum(all_ssim) / N, sum(all_lpips) / N
    #     )
    # )
