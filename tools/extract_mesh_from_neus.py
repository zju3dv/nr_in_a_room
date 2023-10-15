import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"  # noqa
os.environ["MKL_NUM_THREADS"] = "1"  # noqa
sys.path.append(".")  # noqa

import torch
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import mcubes
import open3d as o3d
from plyfile import PlyData, PlyElement
from argparse import ArgumentParser
import time
import plyfile
import skimage
import skimage.measure

# from models.rendering import
from models_neurecon.neus import get_model
from models.image_attributes import ImageAttributes

from utils import load_ckpt

torch.backends.cudnn.benchmark = True


def get_opts():
    parser = ArgumentParser()
    parser.add_argument(
        "--chunk",
        type=int,
        default=32 * 1024,
        help="chunk size to split the input to avoid OOM",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="pretrained checkpoint path to load",
    )
    parser.add_argument(
        "--use_emb_a",
        default=False,
        action="store_true",
        help="appearance embedding",
    )
    parser.add_argument(
        "--N_grid",
        type=int,
        default=256,
        help="size of the grid on 1 side, larger=higher resolution",
    )
    parser.add_argument(
        "--x_range",
        nargs="+",
        type=float,
        default=[-1.5, 1.5],
        help="x range of the object",
    )
    parser.add_argument(
        "--y_range",
        nargs="+",
        type=float,
        default=[-1.5, 1.5],
        help="x range of the object",
    )
    parser.add_argument(
        "--z_range",
        nargs="+",
        type=float,
        default=[-1.5, 1.5],
        help="x range of the object",
    )
    parser.add_argument(
        "--sdf_th",
        type=float,
        default=0.0,
        help="threshold to consider a location is occupied",
    )
    parser.add_argument("--obj_id", type=int, default=0, help="obj_id")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_opts()

    conf = {
        "inside_out": args.obj_id == 0,
        "model": {
            "N_max_objs": 128,
            "N_obj_embedding": 64,
        },
    }

    # with lights
    # conf["model"].update({"N_max_lights": 128, "N_light_embedding": 16})
    conf["model"].update({"N_max_lights": 1024, "N_light_embedding": 16})
    if args.use_emb_a:
        conf["model"].update(
            {"N_max_appearance_frames": 10000, "N_appearance_embedding": 16}
        )

    neus, render_kwargs_train, render_kwargs_test = get_model(
        config_path="config/neus.yaml", need_trainer=False, extra_conf=conf
    )
    image_attributes = ImageAttributes(conf["model"])

    load_ckpt(neus, args.ckpt_path, model_name="neus")
    load_ckpt(image_attributes, args.ckpt_path, model_name="image_attributes")

    neus.cuda().eval()
    image_attributes.cuda().eval()

    # define the dense grid for query
    N = args.N_grid
    xmin, xmax = args.x_range
    ymin, ymax = args.y_range
    zmin, zmax = args.z_range
    # assert xmax-xmin == ymax-ymin == zmax-zmin, 'the ranges must have the same length!'
    x = np.linspace(xmin, xmax, N)
    y = np.linspace(ymin, ymax, N)
    z = np.linspace(zmin, zmax, N)

    xyz_ = torch.FloatTensor(np.stack(np.meshgrid(x, y, z), -1).reshape(-1, 3)).cuda()
    dir_ = torch.zeros_like(xyz_).cuda()
    # sigma is independent of direction, so any value here will produce the same result

    obj_id = args.obj_id

    # predict sigma (occupancy) for each grid location
    print("Predicting occupancy ...")
    with torch.no_grad():
        B = xyz_.shape[0]
        out_chunks = []
        for i in tqdm(range(0, B, args.chunk)):
            xyz_chunk = xyz_[i : i + args.chunk]  # (N, 3)
            N_local_rays = xyz_chunk.shape[0]
            inst_embedded = image_attributes.embedding_instance(
                torch.ones((N_local_rays)).long().cuda() * obj_id
            )
            res_chunk = neus.implicit_surface.forward(
                x=xyz_chunk, obj_code=inst_embedded, return_h=False
            )
            out_chunks += [res_chunk.cpu()]
        sdf = torch.cat(out_chunks, 0)
    # import ipdb; ipdb.set_trace()
    # sdf = sdf[:, -1].cpu().numpy()
    sdf = sdf.numpy().reshape(N, N, N)
    np.save("debug/sdf.npy", sdf)
    # convert_sigma_samples_to_ply(sdf, [-1.5, -1.5, -1.5], 0.1, 'debug/test.ply', level=0)

    # exit(0)
    # perform marching cube algorithm to retrieve vertices and triangle mesh
    print("Extracting mesh ...")
    vertices, triangles = mcubes.marching_cubes(sdf, args.sdf_th)

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

    o3d.io.write_triangle_mesh(f"debug/extracted_neus_{obj_id}.ply", mesh)
    # remove noise in the mesh by keeping only the biggest cluster
    # print('Removing noise ...')
    # mesh = o3d.io.read_triangle_mesh(f"debug/extracted_neus_{obj_id}.ply")
    # idxs, count, _ = mesh.cluster_connected_triangles()
    # max_cluster_idx = np.argmax(count)
    # triangles_to_remove = [
    #     i for i in range(len(face)) if idxs[i] != max_cluster_idx
    # ]
    # mesh.remove_triangles_by_index(triangles_to_remove)
    # mesh.remove_unreferenced_vertices()
    # print(
    #     f'Mesh has {len(mesh.vertices)/1e6:.2f} M vertices and {len(mesh.triangles)/1e6:.2f} M faces.'
    # )
    bbox = mesh.get_axis_aligned_bounding_box()
    print(bbox)
    # import ipdb

    # ipdb.set_trace()
    # o3d.io.write_triangle_mesh("debug/extracted_clean.ply", mesh)
