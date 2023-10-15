import sys
import os

sys.path.append(os.getcwd())  # noqa

import numpy as np
import argparse
import cv2
from tqdm import tqdm
from utils.util import list_dir, read_json

if __name__ == "__main__":
    base_dir = sys.argv[1]
    scenes = list_dir(base_dir)

    curr_progress = 0
    start_progress = 400

    for scene in scenes:
        scene_folder = os.path.join(base_dir, scene)
        if not os.path.isdir(scene_folder):
            continue
        views = list_dir(scene_folder)
        for view in tqdm(views):
            if curr_progress < start_progress:
                curr_progress += 1
                continue
            rgb_file = os.path.join(scene_folder, view, "rgb.png")
            data_json = os.path.join(scene_folder, view, "data.json")
            data_info = read_json(data_json)
            print("progress", curr_progress)
            print(rgb_file)
            print(data_info["name"], data_info["scene"], data_info["room"])
            print("Num objs = ", len(data_info["objs"]))
            img = cv2.imread(rgb_file)
            cv2.imshow("view", img)
            key = cv2.waitKey(0)
            curr_progress += 1
            if key == ord("q"):
                exit(0)
