import sys
import os

sys.path.append(os.getcwd())  # noqa
import ipdb
import numpy as np
import torch
import cv2
import imageio
import time
from tqdm import tqdm
from utils.util import list_dir
from tools.apply_light_map_2d import compute_normal_from_depth

if __name__ == "__main__":
    base_dir = sys.argv[1]
    folders = list_dir(base_dir)
    for folder in tqdm(folders):
        depth_map = np.load(os.path.join(base_dir, folder, "depth.npy"))
        normal_map = compute_normal_from_depth(depth_map, is_panorama=True)
        # convert to png
        normal_map = ((normal_map + 1) / 2 * 255).astype(np.uint8)
        normal_map[depth_map == 0] = 0
        imageio.imwrite(os.path.join(base_dir, folder, "normal.png"), normal_map)
