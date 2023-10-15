import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import sys
import os

sys.path.append(os.getcwd())  # noqa
from utils.util import ensure_dir


def colored_data(x, cmap="jet", d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(x)
    if d_max is None:
        d_max = np.max(x)
    # print(np.min(x), np.max(x))
    x_relative = (x - d_min) / (d_max - d_min)
    cmap_ = plt.cm.get_cmap(cmap)
    return (255 * cmap_(x_relative)[:, :, :3]).astype(np.uint8)  # H, W, C


if __name__ == "__main__":
    base_dir = os.sys.argv[1]
    process_tags = ["seg"]
    output_paths = ["blend"]
    for i in range(len(process_tags)):
        proc = process_tags[i]
        out = output_paths[i]
        ensure_dir(f"{base_dir}/{out}")
        for i in range(3000):
            img_path = f"{base_dir}/full/frame_{i:05d}.png"
            if not os.path.exists(img_path):
                continue
            print("\r{}".format(i), end="")
            print("path", img_path)
            #     print("finish at", i)
            #     break
            inst_seg_path = f"{base_dir}/full/frame_{i:05d}.{proc}.png"
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            inst = cv2.imread(inst_seg_path, cv2.IMREAD_ANYDEPTH)
            color_img = colored_data(inst, d_min=0, d_max=12)
            blend = img // 2 + color_img // 2
            blend_out_path = f"{base_dir}/{out}/{i:05d}.jpg"
            cv2.imwrite(blend_out_path, blend)
