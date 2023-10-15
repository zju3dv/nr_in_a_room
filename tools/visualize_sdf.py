import numpy as np
import open3d as o3d
import argparse
import os
import sys

sys.path.append(os.getcwd())  # noqa

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from skimage import measure
from skimage.draw import ellipsoid
from tools.O3dVisualizer import O3dVisualizer, map_to_color


def ellip_level_set():
    # Generate a level set about zero of two identical ellipsoids in 3D
    ellip_base = ellipsoid(6, 10, 16, levelset=True)
    ellip_double = np.concatenate((ellip_base[:-1, ...], ellip_base[2:, ...]), axis=0)
    return ellip_double


def marching_cube_vis(z):
    # Use marching cubes to obtain the surface mesh of these ellipsoids
    verts, faces, normals, values = measure.marching_cubes(z, 0)

    # Display resulting triangular mesh using Matplotlib. This can also be done
    # with mayavi (see skimage.measure.marching_cubes_lewiner docstring).
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor("k")
    ax.add_collection3d(mesh)

    ax.set_xlabel("x-axis: a = 6 per ellipsoid")
    ax.set_ylabel("y-axis: b = 10")
    ax.set_zlabel("z-axis: c = 16")

    ax.set_xlim(0, 24)  # a = 6 (times two for 2nd ellipsoid)
    ax.set_ylim(0, 20)  # b = 10
    ax.set_zlim(0, 32)  # c = 16

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # params
    parser = argparse.ArgumentParser()
    # data paths
    parser.add_argument("--path", default=None)
    args = parser.parse_args()

    if args.path:
        # input must be matrix with (M, N, K)
        grids = np.load(args.path)
    else:
        grids = ellip_level_set()

    # convert grid to 3d points (N, 3)
    gs = grids.shape
    coords = np.meshgrid(np.arange(gs[0]), np.arange(gs[1]), np.arange(gs[2]))
    coords_ind = np.stack(coords).reshape(3, -1).T

    # get sdf for each points
    zs = grids[coords_ind[:, 0], coords_ind[:, 1], coords_ind[:, 2]]
    points = coords_ind / gs[0]

    # remove positive or negative values
    # https://github.com/facebookresearch/DeepSDF/issues/15
    mask = zs < 0
    zs = zs[mask]
    points = points[mask]

    # random downsample
    N = zs.shape[0]
    rand_ind = np.random.choice(
        np.arange(N), size=min(10000, int(N * 0.05)), replace=False
    )
    zs = zs[rand_ind]
    points = points[rand_ind]

    # visualize
    visualizer = O3dVisualizer()
    colors = map_to_color(-zs)
    visualizer.add_np_points(points, size=0.001, color=colors)

    visualizer.run_visualize()
