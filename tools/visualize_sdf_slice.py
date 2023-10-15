import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import argparse
import os
import sys

sys.path.append(os.getcwd())  # noqa

if __name__ == "__main__":
    # params
    parser = argparse.ArgumentParser()
    # data paths
    parser.add_argument("--path", default=None)
    args = parser.parse_args()

    grids = np.load(args.path)

    # sdf_slice = grids[128, :, :]
    # sdf_slice = grids[128, :, :]
    sdf_slice = grids[:, 128, :]

    plt.imshow(sdf_slice, cmap="coolwarm")
    plt.colorbar()
    plt.show()
