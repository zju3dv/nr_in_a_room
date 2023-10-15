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

# from models.rendering import
from models.nerf_object import NeRF_Object, Embedding
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
        "--sigma_threshold",
        type=float,
        default=20.0,
        help="threshold to consider a location is occupied",
    )
    parser.add_argument("--obj_id", type=int, default=0, help="obj_id")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_opts()

    embedding_xyz = Embedding(3, 10)
    embedding_dir = Embedding(3, 4)
    embeddings = {"xyz": embedding_xyz, "dir": embedding_dir}
    conf = {
        "N_max_objs": 128,
        "N_obj_embedding": 64,
    }
    nerf_fine = NeRF_Object(conf)
    image_attributes = ImageAttributes(conf)

    load_ckpt(nerf_fine, args.ckpt_path, model_name="nerf_fine")
    load_ckpt(image_attributes, args.ckpt_path, model_name="image_attributes")

    nerf_fine.cuda().eval()
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
            xyz_embedded = embedding_xyz(
                xyz_[i : i + args.chunk]
            )  # (N, embed_xyz_channels)
            N_local_rays = xyz_embedded.shape[0]
            inst_embedded = image_attributes.embedding_instance(
                torch.ones((N_local_rays)).long().cuda() * obj_id
            )
            input_dict = {
                "xyz_embedded": xyz_embedded,
                "inst_embedded": inst_embedded,
            }
            out_chunks += [nerf_fine.forward_instance_mask(input_dict)]
        rgbsigma = torch.cat(out_chunks, 0)

    sigma = rgbsigma[:, -1].cpu().numpy()
    sigma = np.maximum(sigma, 0).reshape(N, N, N)

    # perform marching cube algorithm to retrieve vertices and triangle mesh
    print("Extracting mesh ...")
    vertices, triangles = mcubes.marching_cubes(sigma, args.sigma_threshold)

    ##### Until mesh extraction here, it is the same as the original repo. ######

    vertices_ = (vertices / N).astype(np.float32)
    ## invert x and y coordinates (WHY? maybe because of the marching cubes algo)
    x_ = (ymax - ymin) * vertices_[:, 1] + ymin
    y_ = (xmax - xmin) * vertices_[:, 0] + xmin
    vertices_[:, 0] = x_
    vertices_[:, 1] = y_
    vertices_[:, 2] = (zmax - zmin) * vertices_[:, 2] + zmin
    vertices_.dtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]

    face = np.empty(len(triangles), dtype=[("vertex_indices", "i4", (3,))])
    face["vertex_indices"] = triangles

    PlyData(
        [
            PlyElement.describe(vertices_[:, 0], "vertex"),
            PlyElement.describe(face, "face"),
        ]
    ).write(f"debug/extracted_{obj_id}.ply")

    # remove noise in the mesh by keeping only the biggest cluster
    # print('Removing noise ...')
    mesh = o3d.io.read_triangle_mesh(f"debug/extracted_{obj_id}.ply")
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
    # o3d.io.write_triangle_mesh("debug/extracted_clean.ply", mesh)
