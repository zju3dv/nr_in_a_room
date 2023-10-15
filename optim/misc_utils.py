import os
import sys

sys.path.append(os.getcwd())  # noqa
import numpy as np
import cv2
import torch
from tqdm import tqdm
import open3d as o3d
import mcubes
from omegaconf import OmegaConf
from typing import List, Optional, Any, Dict, Union
from datasets.geo_utils import rotation_6d_to_matrix
from utils.util import remove_stdout_line, write_json, read_json
from scipy.spatial.transform import Rotation


def read_dataset_config_file(dataset_config_file):
    # load user file
    conf_dataset_file = OmegaConf.load(dataset_config_file)
    # load parent config
    if "parent_config" in conf_dataset_file:
        conf_parent_file = OmegaConf.load(conf_dataset_file.parent_config)
    else:
        conf_parent_file = OmegaConf.create()
    # default config
    conf_default = OmegaConf.load("config/default_conf.yml")
    # merge conf with the priority
    conf_merged = OmegaConf.merge(conf_default, conf_parent_file, conf_dataset_file)
    return conf_merged


def read_testing_config():
    conf_cli = OmegaConf.from_cli()
    conf_test_file = OmegaConf.load(conf_cli.config)
    # read dataset config
    conf_test_file["dataset_config"] = read_dataset_config_file(
        conf_test_file["dataset_config_path"]
    )
    conf_test_file["bg_dataset_config"] = read_dataset_config_file(
        conf_test_file["bg_dataset_config_path"]
    )

    # processing ckpt
    ckpt_path_dict = {}
    for item in conf_test_file["ckpt_lists"]:
        path = item["path"]
        obj_ids = item["obj_ids"]
        neus_conf = item.get("neus_conf", "config/neus.yaml")
        for obj_id in obj_ids:
            ckpt_path_dict[str(obj_id)] = {"path": path, "neus_conf": neus_conf}
    conf_test_file["ckpt_path_dict"] = ckpt_path_dict

    conf_merged = OmegaConf.merge(conf_test_file, conf_cli)
    return conf_merged


def get_object_meta_info(ig_data_base_dir, meta, obj_id):
    """
    Return:
        T_wo: from object to world
        bbox3d: bounding box info
    """
    for obj_info in meta["objs"]:
        if obj_info["id"] == obj_id:
            # load pose info from bounding box info
            bbox3d = obj_info["bdb3d"]
            ret_dict = {"bbox3d": bbox3d}
            if "centroid" in bbox3d and "basis" in bbox3d:
                center = np.array(bbox3d["centroid"])
                rotation = np.array(bbox3d["basis"]).reshape(3, 3)
                T_wo = np.eye(4)
                T_wo[:3, :3] = rotation
                T_wo[:3, 3] = center
                ret_dict["gt_T_wo"] = T_wo
            return ret_dict
    raise RuntimeError("object with id {} not exist in metadata".format(obj_id))


def get_instance_mask(instance_path, img_wh, instance_id=None):
    instance = cv2.resize(
        cv2.imread(instance_path, cv2.IMREAD_ANYDEPTH),
        img_wh,
        interpolation=cv2.INTER_NEAREST,
    )
    if instance_id is None:
        return instance
    else:
        mask = instance == instance_id
        return mask


def get_mask_bbox(mask):
    # crop image
    true_indices = np.nonzero(mask)
    min_h, min_w = np.min(true_indices[0]), np.min(true_indices[1])
    max_h, max_w = np.max(true_indices[0]), np.max(true_indices[1])
    # print(min_h, min_w)
    # print(max_h, max_w)
    # img = img[min_h:max_h+1,min_w:max_w+1,:]
    return min_h, max_h, min_w, max_w


def seg_mask_to_box_mask(mask):
    min_h, max_h, min_w, max_w = get_mask_bbox(mask)
    mask[min_h : max_h + 1, min_w : max_w + 1] = 1
    return mask


def dump_optimization_meta_to_file(filepath: str, obj_pose_dict: Dict[str, Any]):
    object_meta_dict = {}
    for k, v in obj_pose_dict.items():
        Two = np.eye(4)
        Two[:3, 3] = v["trans"].detach().cpu().numpy()
        Two[:3, :3] = rotation_6d_to_matrix(v["rot6d"]).detach().cpu().numpy()
        object_meta_dict[k] = {"Two": Two.tolist()}
    write_json(object_meta_dict, filepath)


def pano_sample_probability(h, w):
    # porportional to sin
    h_prob = np.sin(np.linspace(0, np.pi, h))
    prob = np.tile(h_prob, (w, 1)).T
    return prob


def detect_keypoints(img, circle_radius=5):
    # import ipdb
    # ipdb.set_trace()
    img = (img * 255).astype(np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, 200, 0.01, 10)
    corners = np.int0(corners)
    mask = np.zeros_like(gray)
    for i in corners:
        x, y = i.ravel()
        cv2.circle(mask, center=(x, y), radius=circle_radius, color=255, thickness=-1)
        cv2.circle(img, center=(x, y), radius=circle_radius, color=255, thickness=-1)
    # cv2.imwrite("debug/detect_mask.png", mask)
    # cv2.imwrite("debug/img_with_detection.png", img)
    return mask


def adjust_learning_rate(initial_lr, optimizer, iter, base=0.1, adjust_lr_every=400):
    lr = initial_lr * ((base) ** (iter // adjust_lr_every))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def extract_mesh_from_neus(
    neus_model,
    image_attributes,
    obj_id: int,
    x_range=[-1.5, 1.5],
    y_range=[-1.5, 1.5],
    z_range=[-1.5, 1.5],
    N_grid=256,
    chunk=32 * 1024,
    sdf_th=0,
):
    # define the dense grid for query
    N = N_grid
    xmin, xmax = x_range
    ymin, ymax = y_range
    zmin, zmax = z_range
    # assert xmax-xmin == ymax-ymin == zmax-zmin, 'the ranges must have the same length!'
    x = np.linspace(xmin, xmax, N)
    y = np.linspace(ymin, ymax, N)
    z = np.linspace(zmin, zmax, N)

    xyz_ = torch.FloatTensor(np.stack(np.meshgrid(x, y, z), -1).reshape(-1, 3)).cuda()
    # dir_ = torch.zeros_like(xyz_).cuda()
    # sigma is independent of direction, so any value here will produce the same result

    # predict sigma (occupancy) for each grid location
    # print("Predicting occupancy ...")
    with torch.no_grad():
        B = xyz_.shape[0]
        out_chunks = []
        for i in tqdm(range(0, B, chunk)):
            xyz_chunk = xyz_[i : i + chunk]  # (N, 3)
            N_local_rays = xyz_chunk.shape[0]
            inst_embedded = image_attributes.embedding_instance(
                torch.ones((N_local_rays)).long().cuda() * obj_id
            )
            res_chunk = neus_model.implicit_surface.forward(
                x=xyz_chunk, obj_code=inst_embedded, return_h=False
            )
            out_chunks += [res_chunk.cpu()]
        sdf = torch.cat(out_chunks, 0)
    sdf = sdf.numpy().reshape(N, N, N)
    vertices, triangles = mcubes.marching_cubes(sdf, sdf_th)
    remove_stdout_line(1)

    ##### Until mesh extraction here, it is the same as the original repo. ######

    vertices_ = (vertices / N).astype(np.float64)
    ## invert x and y coordinates (WHY? maybe because of the marching cubes algo)
    x_ = (ymax - ymin) * vertices_[:, 1] + ymin
    y_ = (xmax - xmin) * vertices_[:, 0] + xmin
    vertices_[:, 0] = x_
    vertices_[:, 1] = y_
    vertices_[:, 2] = (zmax - zmin) * vertices_[:, 2] + zmin

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices_.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(triangles.astype(np.int32))
    return mesh


def decompose_to_sRT(Trans):
    t = Trans[:3, 3]
    R = Trans[:3, :3]
    # assume x y z have the same scale
    scale = np.linalg.norm(R[:3, 0])
    R = R / scale
    return scale, R, t


def read_real_scene_localization(pose_path: str, transform_info_json_path: str):
    pose_dict = {}
    transform_info = read_json(transform_info_json_path)
    trans_colmap_to_arkit = np.array(transform_info["transform_colmap_to_arkit_sRT"])
    trans_align = np.array(transform_info["transform_alignment"])
    with open(pose_path) as file:
        lines = file.readlines()
        lines = lines[1:]
        for line in lines:
            fname, tx, ty, tz, qx, qy, qz, qw, _, _ = line.strip().split(" ")
            fname += ".png"
            pose = np.eye(4)
            pose[0, 3] = tx
            pose[1, 3] = ty
            pose[2, 3] = tz
            # Twc
            pose[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            # pose = np.linalg.inv(pose)
            # pose_ndc = np.linalg.inv(pose_ndc)

            # convert to ndc
            # pose_ndc = pose
            # fix_rot = np.array([1, 0, 0, 0, -1, 0, 0, 0, -1]).reshape(3, 3)
            # pose_ndc[:3, :3] = pose_ndc[:3, :3] @ fix_rot

            # transform to arkit pose
            s, R, t = decompose_to_sRT(trans_colmap_to_arkit)
            # pose_ndc = transform_colmap_to_arkit @ pose_ndc
            # print(s, R, t)
            pose[:3, 3] = R @ (pose[:3, 3] * s) + t
            pose[:3, :3] = R @ pose[:3, :3]

            # apply alignment to poses
            pose = trans_align @ pose

            pose_dict[fname] = {"pose_slam_Twc": pose}
            # print(fname, pose)
    return pose_dict


def read_real_scene_localization_with_name(arrangement_name):
    localization_info = read_real_scene_localization(
        f"data/real_room_0/arrangement_panorama_select/{arrangement_name}/traj.txt",
        "data/real_room_0/objects/000/background_hloc_neus_normal_converge/transform_info.json",
    )
    return localization_info


if __name__ == "__main__":
    loc_info = read_real_scene_localization_with_name("arrangement3")
    for k, v in loc_info.items():
        Twc = v["pose_slam_Twc"]
        eye = Twc[:3, 3]
        target = np.array([0, 0, 1])
        target = Twc[:3, 3] + Twc[:3, :3] @ target
        up = np.array([0, -1, 0])
        up = Twc[:3, :3] @ up
        loc_info[k]["pose_slam_Twc"] = v["pose_slam_Twc"].tolist()
        camera_dict = {
            "pos": eye.tolist(),
            "target": target.tolist(),
            "up": up.tolist(),
            "world2cam3d": np.linalg.inv(Twc).tolist(),
            "cam3d2world": Twc.tolist(),
            "width": 1024,
            "height": 512,
        }
        loc_info[k]["camera"] = camera_dict
    write_json(loc_info, "arrangement3_loc.json")
