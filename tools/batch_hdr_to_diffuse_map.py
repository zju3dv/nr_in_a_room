import sys
import os

sys.path.append(os.getcwd())  # noqa
import numpy as np
import cv2
import imageio
from tqdm import tqdm

from tools.spherical_harmonics import getDiffuseMap
from utils.util import list_dir

if __name__ == "__main__":
    input_dir = "data/hdr_galary_100"
    output_dir = "debug/hdr_galary_100_diffuse"
    os.makedirs(output_dir, exist_ok=True)

    hdr_files = list_dir(input_dir)

    for hdr_file in tqdm(hdr_files):
        diffuse_map = getDiffuseMap(os.path.join(input_dir, hdr_file))
        imageio.imwrite(
            os.path.join(output_dir, hdr_file), diffuse_map.astype(np.float32)
        )
