import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="path to image")
    parser.add_argument("image_file", type=str)
    args = parser.parse_args()
    img = cv2.imread(args.image_file, cv2.IMREAD_ANYDEPTH)
    # img = (img >> 3) | (img << 13) # for ig background
    img = img * 1e-3
    # img = plt.imread(args.image_file)
    print(img, np.percentile(np.unique(img), 0.1), np.percentile(img, 99.9))

    # conv to show keypoint depth clearly
    # kernel = np.ones((7, 7))
    # img = cv2.filter2D(img, -1, kernel)
    # print(np.max(img))
    plt.imshow(img, cmap="viridis", vmin=0, vmax=5)
    plt.colorbar()
    plt.imsave("depth_vis.png", img, cmap="viridis", vmin=0, vmax=5)
    plt.show()
