import os
import sys

sys.path.append(".")  # noqa
import argparse
import numpy as np
import json
import imageio
import glob
from pyquaternion import Quaternion
from run_colmap_posed import main


def convert_arkit_raw_dict_to_pinhole_dict(arkit_raw_dir, pinhole_dict_file, img_dir):
    print("Writing pinhole_dict to: ", pinhole_dict_file)

    all_image_files = sorted(glob.glob(arkit_raw_dir + "/frame_*.jpg"))

    # with open(cam_dict_file) as fp:
    #     cam_dict = json.load(fp)

    pinhole_dict = {}
    for idx, img_name in enumerate(all_image_files):
        # data_item = json
        img_name = os.path.basename(img_name)
        with open(os.path.join(arkit_raw_dir, img_name[:-3] + "json")) as fp:
            frame_info = json.load(fp)
        if idx == 0:
            im = imageio.imread(os.path.join(img_dir, img_name))
            h, w = im.shape[:2]

        intrinsics = np.array(frame_info["intrinsics"])
        focal, cx, cy = intrinsics[0], intrinsics[2], intrinsics[5]
        K = np.eye(4)
        K[0, 0] = K[1, 1] = focal
        K[0, 2] = cx
        K[1, 2] = cy
        pose = np.array(frame_info["cameraPoseARFrame"]).reshape(4, 4)
        fix_rot = np.array([1, 0, 0, 0, -1, 0, 0, 0, -1]).reshape(3, 3)
        # pose: SLAM Twc with xyz-> right down forwawrd
        pose[:3, :3] = pose[:3, :3] @ fix_rot
        # to Tcw
        W2C = np.linalg.inv(pose)

        # params
        fx = K[0, 0]
        fy = K[1, 1]
        assert np.isclose(K[0, 1], 0.0)
        cx = K[0, 2]
        cy = K[1, 2]

        print(img_name)
        R = W2C[:3, :3]
        print(R)
        u, s_old, vh = np.linalg.svd(R, full_matrices=False)
        s = np.round(s_old)
        print("s: {} ---> {}".format(s_old, s))
        R = np.dot(u * s, vh)

        qvec = Quaternion(matrix=R)
        tvec = W2C[:3, 3]

        params = [
            w,
            h,
            fx,
            fy,
            cx,
            cy,
            qvec[0],
            qvec[1],
            qvec[2],
            qvec[3],
            tvec[0],
            tvec[1],
            tvec[2],
        ]
        pinhole_dict[img_name] = params

    with open(pinhole_dict_file, "w") as fp:
        json.dump(pinhole_dict, fp, indent=2, sort_keys=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arkit_raw_dir",
        default="/home/ybbbbt/Developer/neural_scene/data/arkit_recon/arkit_box_2",
    )
    parser.add_argument(
        "--recon_workspace",
        default="/home/ybbbbt/Developer/neural_scene/data/arkit_recon/arkit_box_2_colmap",
        help="workspace should contain image folder",
    )
    args = parser.parse_args()
    arkit_raw_dir = args.arkit_raw_dir
    # should contain "images" folder
    recon_workspace = args.recon_workspace
    img_dir = os.path.join(recon_workspace, "images")

    os.makedirs(recon_workspace, exist_ok=True)
    pinhole_dict_file = os.path.join(recon_workspace, "pinhole_dict.json")
    convert_arkit_raw_dict_to_pinhole_dict(arkit_raw_dir, pinhole_dict_file, img_dir)

    run_mvs = False

    # main(img_dir, pinhole_dict_file, recon_workspace, run_mvs, refine_with_gba=False)
    main(img_dir, pinhole_dict_file, recon_workspace, run_mvs)
