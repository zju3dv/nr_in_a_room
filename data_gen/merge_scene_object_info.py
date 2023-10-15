import sys
import os

sys.path.append(os.getcwd())  # noqa
import ipdb
import numpy as np
from tqdm import tqdm
from utils.util import list_dir, read_json, write_json

if __name__ == "__main__":
    # scene_src = (
    #     "data/Beechwood_1_int_bedroom_0_no_random/scene_multi_lights/probe_03"
    # )
    # merge_dir = "data/Beechwood_1_int_bedroom_0_no_random/scene/full"
    scene_src = sys.argv[1]
    merge_dir = sys.argv[2]
    os.makedirs(merge_dir, exist_ok=True)

    scene_folders = list_dir(scene_src)

    room_info = {}
    exist_obj_ids = set()

    for idx, scene_folder in tqdm(enumerate(scene_folders)):
        if scene_folder == "full":
            continue
        scene_info = read_json(os.path.join(scene_src, scene_folder, "data.json"))
        if idx == 0:
            room_info = scene_info
            for obj_info in room_info["objs"]:
                exist_obj_ids.add(obj_info["id"])
        else:
            for obj_info in scene_info["objs"]:
                obj_id = obj_info["id"]
                if obj_id not in exist_obj_ids:
                    room_info["objs"].append(obj_info)
                    exist_obj_ids.add(obj_id)

    write_json(room_info, os.path.join(merge_dir, "data.json"))
