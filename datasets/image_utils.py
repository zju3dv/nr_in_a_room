from matplotlib import pyplot as plt
import numpy as np
import cv2
from PIL import Image
import os


def rebalance_mask(mask, fg_weight=None, bg_weight=None):
    if fg_weight is None and bg_weight is None:
        foreground_cnt = max(mask.sum(), 1)
        background_cnt = max((~mask).sum(), 1)
        balanced_weight = np.ones_like(mask).astype(np.float32)
        balanced_weight[mask] = float(background_cnt) / foreground_cnt
        balanced_weight[~mask] = float(foreground_cnt) / background_cnt
    else:
        balanced_weight = np.ones_like(mask).astype(np.float32)
        balanced_weight[mask] = fg_weight
        balanced_weight[~mask] = bg_weight
    # print('fg {} bg {}'.format(foreground_cnt, background_cnt))
    # print(balanced_weight.min(), balanced_weight.max())
    # print(balanced_weight.shape)
    # cv2.normalize(balanced_weight, balanced_weight, 0, 1.0, cv2.NORM_MINMAX)
    # cv2.imshow('img_balanced_weight', balanced_weight)
    # cv2.waitKey(5)
    return balanced_weight


def compute_distance_transfrom_weights(
    mask, uncertain_pixel_distance=15, fg_bg_balance_weight=False
):
    instance_mask = mask
    max_dist = uncertain_pixel_distance
    dt_field = np.zeros_like(instance_mask, dtype=np.uint8)
    dt_field[instance_mask] = 255
    dist1 = cv2.distanceTransform(dt_field, cv2.DIST_L2, 3)

    dt_field_inv = np.zeros_like(instance_mask, dtype=np.uint8)
    dt_field_inv[~instance_mask] = 255
    dist2 = cv2.distanceTransform(dt_field_inv, cv2.DIST_L2, 3)

    dist_combine = np.ones_like(dist1) * max_dist

    dist1[dist1 > max_dist] = max_dist
    dist2[dist2 > max_dist] = max_dist

    dist1_mask = (dist1 < max_dist) * (dist1 > 0)
    dist_combine[dist1_mask] = dist1[dist1_mask]
    disk2_mask = (dist2 < max_dist) * (dist2 > 0)
    dist_combine[disk2_mask] = dist2[disk2_mask]

    cv2.normalize(dist_combine, dist_combine, 0, 1.0, cv2.NORM_MINMAX)

    if fg_bg_balance_weight:
        dist_combine *= rebalance_mask(mask)

    # cv2.normalize(dist_combine, dist_combine, 0, 1, cv2.NORM_MINMAX)
    # cv2.imshow('Distance Transform Image', dist_combine)
    # cv2.waitKey(5)
    return dist_combine


def variance_of_laplacian(image: np.ndarray) -> np.ndarray:
    """Compute the variance of the Laplacian which measure the focus."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def filter_blur_images(root_dir, frames, blur_filter_perc, dump_images=False):
    blur_filter_perc = 85.0  # @param {type: 'number'}
    if blur_filter_perc > 0.0:
        print("Loading images.")
        images = []
        for frame in frames:
            img = cv2.imread(os.path.join(root_dir, f"{frame['file_path']}.png"))
            img = cv2.resize(img, (640, 480))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img = Image.open(os.path.join(root_dir, f"{frame['file_path']}.png"))
            images += [img]
        print("Computing blur scores.")
        blur_scores = np.array([variance_of_laplacian(im) for im in images])
        print(blur_scores)
        blur_thres = np.percentile(blur_scores, blur_filter_perc)
        blur_filter_inds = np.where(blur_scores >= blur_thres)[0]
        blur_filter_scores = [blur_scores[i] for i in blur_filter_inds]
        blur_filter_inds = blur_filter_inds[np.argsort(blur_filter_scores)]
        blur_filter_scores = np.sort(blur_filter_scores)

        plt.figure(figsize=(15, 10))
        plt.subplot(121)
        plt.title("Least blurry")
        plt.imshow(images[blur_filter_inds[-1]])
        plt.subplot(122)
        plt.title("Most blurry")
        plt.imshow(images[blur_filter_inds[0]])
        plt.show()
