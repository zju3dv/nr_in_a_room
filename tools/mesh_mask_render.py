import sys
import os
import pyglet

sys.path.append(os.getcwd())  # noqa
pyglet.options["shadow_window"] = False
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from utils.util import read_json

from pyrender import (
    PerspectiveCamera,
    DirectionalLight,
    SpotLight,
    PointLight,
    MetallicRoughnessMaterial,
    Primitive,
    Mesh,
    Node,
    Scene,
    Viewer,
    OffscreenRenderer,
    RenderFlags,
)


if __name__ == "__main__":
    # obj_trimesh = trimesh.load(
    #     "/home/ybbbbt/Developer/neural_scene/data/arkit_recon/arkit_red_face/textured_output_clean_v2.ply"
    # )

    obj_trimesh = trimesh.load(
        # "/home/ybbbbt/Developer/neural_scene/data/arkit_recon/arkit_red_face/textured_output.obj"
        "/home/ybbbbt/Developer/neural_scene/transformed.obj"
    )

    obj_mesh = Mesh.from_trimesh(obj_trimesh)

    frame_info = read_json(
        # "/home/ybbbbt/Developer/neural_scene/data/arkit_recon/arkit_red_face/frame_00000.json"
        "/home/ybbbbt/Developer/neural_scene/data/arkit_recon/arkit_nightstand_2/frame_00102.json"
    )
    pose_ndc = np.array(frame_info["cameraPoseARFrame"]).reshape(4, 4)
    from scipy.spatial.transform import Rotation as R

    pose_ndc[:3, :3] = R.from_quat(
        [0.0700325, 0.530018, 0.198129, 0.821536]
    ).as_matrix()
    pose_ndc[:3, 3] = np.array([-0.501506, -1.67429, 3.14581])
    pose_ndc = np.linalg.inv(pose_ndc)
    fix_rot = np.array([1, 0, 0, 0, -1, 0, 0, 0, -1]).reshape(3, 3)
    pose_ndc[:3, :3] = pose_ndc[:3, :3] @ fix_rot

    intrinsics = np.array(frame_info["intrinsics"])
    focal, cx, cy = intrinsics[0], intrinsics[2], intrinsics[5]

    print(focal, cx, cy)

    yfov = np.arctan(cy / focal) * 2

    print("yfov =", yfov)

    # cam = PerspectiveCamera(yfov=(np.pi / 3))
    cam = PerspectiveCamera(yfov=yfov)
    # cam_pose = np.eye(4)
    # cam_pose = np.array(
    #     [
    #         [0.0, -np.sqrt(2) / 2, np.sqrt(2) / 2, 0.5],
    #         [1.0, 0.0, 0.0, 0.0],
    #         [0.0, np.sqrt(2) / 2, np.sqrt(2) / 2, 0.4],
    #         [0.0, 0.0, 0.0, 1.0],
    #     ]
    # )
    cam_pose = pose_ndc

    scene = Scene(ambient_light=np.array([0.5, 0.5, 0.5, 1.0]))
    obj_node = Node(mesh=obj_mesh, translation=np.zeros(3))
    scene.add_node(obj_node)
    cam_node = scene.add(cam, pose=cam_pose)
    r = OffscreenRenderer(viewport_width=640, viewport_height=480)
    color, depth = r.render(scene)

    plt.figure()
    plt.imshow(color)
    plt.imsave("color.png", color)
    plt.show()

    plt.figure()
    plt.imshow(depth)
    plt.imsave("depth.png", depth)
    plt.show()

    nm = {node: 20 * (i + 1) for i, node in enumerate(scene.mesh_nodes)}
    seg = r.render(scene, RenderFlags.SEG, nm)[0]
    plt.figure()
    plt.imshow(seg)
    plt.show()

    r.delete()
