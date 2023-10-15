import os, sys

sys.path.append(os.getcwd())  # noqa
from utils.util import ensure_dir, write_json
from habitat_sim import sensor
from habitat_sim.agent.agent import AgentState

from tqdm import tqdm
import magnum as mn
import numpy as np
from matplotlib import is_interactive, pyplot as plt
import cv2

import habitat_sim
from habitat_sim.gfx import LightInfo, LightPositionModel
from habitat_sim.utils.common import quat_from_angle_axis
import quaternion
from scipy.spatial.transform import Rotation as R

dir_path = os.path.dirname(os.path.realpath(__file__))
output_path = os.path.join(dir_path, "gen_output/")
# object_data_path = '/home/ybbbbt/Data/habitat_data/google_scanned_objects/Nintendo_Mario_Action_Figure/meshes/model.obj'
object_data_path = "data/google_scanned_objects/Crunch_Girl_Scouts_Candy_Bars_Peanut_Butter_Creme_78_oz_box/meshes/model.obj"
scene_data_path = (
    "data/official_data/scene_datasets/habitat-test-scenes/skokloster-castle.glb"
)

# settings
render_frame_num = 150
cam_height = 1.5
cam_radius = 0.7
obj_scale = 5.0
rotate_per_frame = 15
downward_angle_begin = 70
# downward_angle_end = 20
downward_angle_end = 45
camera_angle_x = 80
# place_pos = np.array([-0.360426, 0.7, 16.304])
# meshlab (x, y, z) -> (x, z, -y)
place_pos = np.array([-1.6, 0.7, 9.0])

is_equirectangular = True

sensor_settings = {
    "color_sensor": True,
    "depth_sensor": True,
    "semantic_sensor": True,
    "hfov": camera_angle_x,
    # "resolution": [960, 1080],
    "resolution": [480, 640],
}

if is_equirectangular:
    sensor_settings["resolution"] = [960, 1920]


def spheric_pose(theta, phi, radius, height):
    trans_t = lambda t: np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, -0.9 * t],
            [0, 0, 1, t],
            [0, 0, 0, 1],
        ]
    )

    rot_phi = lambda phi: np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(phi), -np.sin(phi), 0],
            [0, np.sin(phi), np.cos(phi), 0],
            [0, 0, 0, 1],
        ]
    )

    rot_theta = lambda th: np.array(
        [
            [np.cos(th), 0, -np.sin(th), 0],
            [0, 1, 0, 0],
            [np.sin(th), 0, np.cos(th), 0],
            [0, 0, 0, 1],
        ]
    )

    c2w = rot_theta(theta) @ rot_phi(phi) @ trans_t(radius)
    c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
    c2w[2, 3] += height
    return c2w[:3]


def create_spheric_poses(radius, downward_deg, height, n_poses):
    """
    Create circular poses around z axis.
    Inputs:
        radius: the (negative) height and the radius of the circle.

    Outputs:
        spheric_poses: (n_poses, 3, 4) the poses in the circular path
    """
    spheric_poses = []
    for th in np.linspace(0, 2 * np.pi, n_poses + 1)[:-1]:
        spheric_poses += [
            spheric_pose(th, -(downward_deg * np.pi / 180), radius, height)
        ]  # 36 degree view downwards
    return np.stack(spheric_poses, 0)


def random_rot(upper_view=True):
    if upper_view:
        rot = np.random.uniform(0, 1, size=3) * (1, 0, 2 * np.pi)
        rot[0] = np.abs(np.arccos(1 - 2 * rot[0]) - np.pi / 2)
    else:
        rot = np.random.uniform(0, 2 * np.pi, size=3)
    # return R.from_euler('zyx', rot, degrees=False).as_matrix()
    return R.from_euler("yxz", rot, degrees=False).as_matrix()


def show_save_img(data, show, save, sensor_type, idx):
    plt.figure(0, figsize=(6, 6))
    # if sensor_type == 'rgb_3rd_person':
    plt.imshow(data, interpolation="nearest")
    plt.axis("off")
    if save:
        if sensor_type == "color_sensor":
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f"{output_path}/full/{idx:05d}.png", data)
        elif sensor_type == "depth_sensor":
            cv2.imwrite(
                f"{output_path}/full/{idx:05d}.depth.png",
                (data * 1000).astype(np.uint16),
            )
        elif sensor_type == "semantic_sensor":
            cv2.imwrite(
                f"{output_path}/full/{idx:05d}.instance.png", data.astype(np.uint16)
            )
    if show:
        plt.show(block=False)
        plt.pause(0.1)


def get_obs(sim, show, save, idx):
    sensor_names = ["color_sensor", "depth_sensor", "semantic_sensor"]
    for sensor_name in sensor_names:
        obs = sim.get_sensor_observations()[sensor_name]
        show_save_img(obs, show, save, sensor_name, idx)
    return obs


def place_agent(sim, rot, position):
    # place our agent in the scene
    agent_state = habitat_sim.AgentState()
    # agent_state.position = [5.0, 0.0, 1.0]
    agent_state.position = position
    # agent_state.rotation = quat_from_angle_axis(
    #     math.radians(70), np.array([0, 1.0, 0])
    # ) * quat_from_angle_axis(math.radians(-20), np.array([1.0, 0, 0]))
    # import ipdb; ipdb.set_trace()
    if isinstance(rot, np.ndarray):
        rot = quaternion.from_rotation_matrix(rot)
    agent_state.rotation = rot
    if len(sim.agents) == 0:
        agent = sim.initialize_agent(0, agent_state)
    else:
        agent = sim.agents[0]
        sim.agents[0].set_state(agent_state)
    return agent


def make_configuration():
    # simulator configuration
    backend_cfg = habitat_sim.SimulatorConfiguration()
    # backend_cfg.scene_id = "data/scene_datasets/habitat-test-scenes/van-gogh-room.glb"
    backend_cfg.scene_id = scene_data_path
    backend_cfg.enable_physics = True
    sensor_specs = []

    if sensor_settings["color_sensor"]:
        # agent configuration
        sensor_cfg = (
            habitat_sim.EquirectangularSensorSpec()
            if is_equirectangular
            else habitat_sim.CameraSensorSpec()
        )
        sensor_cfg.resolution = sensor_settings["resolution"]
        sensor_cfg.sensor_type = habitat_sim.SensorType.COLOR
        sensor_cfg.position = [0, 0, 0]
        sensor_cfg.hfov = sensor_settings["hfov"]
        sensor_cfg.uuid = "color_sensor"
        sensor_specs.append(sensor_cfg)

    if sensor_settings["depth_sensor"]:
        # agent configuration
        sensor_cfg = (
            habitat_sim.EquirectangularSensorSpec()
            if is_equirectangular
            else habitat_sim.CameraSensorSpec()
        )
        sensor_cfg.resolution = sensor_settings["resolution"]
        sensor_cfg.sensor_type = habitat_sim.SensorType.DEPTH
        sensor_cfg.position = [0, 0, 0]
        sensor_cfg.hfov = sensor_settings["hfov"]
        sensor_cfg.uuid = "depth_sensor"
        sensor_specs.append(sensor_cfg)

    if sensor_settings["semantic_sensor"]:
        # agent configuration
        sensor_cfg = (
            habitat_sim.EquirectangularSensorSpec()
            if is_equirectangular
            else habitat_sim.CameraSensorSpec()
        )
        sensor_cfg.resolution = sensor_settings["resolution"]
        sensor_cfg.sensor_type = habitat_sim.SensorType.SEMANTIC
        sensor_cfg.position = [0, 0, 0]
        sensor_cfg.hfov = sensor_settings["hfov"]
        sensor_cfg.uuid = "semantic_sensor"
        sensor_specs.append(sensor_cfg)

    # sensor_cfg = habitat_sim.CameraSensorSpec()
    # sensor_cfg.resolution = sensor_settings["resolution"]
    # sensor_cfg.sensor_type = habitat_sim.SensorType.COLOR
    # sensor_cfg.position = [0.0, 1.0, 0.3]
    # sensor_cfg.position = [0, 0, 0]
    # sensor_cfg.orientation = [-45, 0.0, 0.0]
    # sensor_cfg.uuid = "rgb_3rd_person"
    # sensor_specs.append(sensor_cfg)

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs

    return habitat_sim.Configuration(backend_cfg, [agent_cfg])


# [/setup]


# This is wrapped such that it can be added to a unit test
def main(show_imgs=True, save_imgs=False):
    ensure_dir(output_path)
    ensure_dir(os.path.join(output_path, "full"))

    # [default scene lighting]

    # create the simulator and render flat shaded scene
    cfg = make_configuration()
    sim = habitat_sim.Simulator(cfg)

    # get_obs(sim, show_imgs, save_imgs)

    # get the rigid object attributes manager, which manages
    # templates used to create objects
    obj_template_mgr = sim.get_object_template_manager()
    # get the rigid object manager, which provides direct
    # access to objects
    rigid_obj_mgr = sim.get_rigid_object_manager()

    # load a custom object
    box_template = habitat_sim.attributes.ObjectAttributes()
    box_template.render_asset_handle = object_data_path
    # box_template.scale = np.array([1.0, 1.0, 1.0])
    box_template.scale = np.array([obj_scale, obj_scale, obj_scale])
    box_template.orient_up = (0.0, 0.0, 1.0)
    box_template.orient_front = (1.0, 0.0, 0.0)
    # set the default semantic id for this object template
    box_template.semantic_id = 10  # @param{type:"integer"}
    box_template_id = obj_template_mgr.register_template(box_template, "box")
    box_obj = rigid_obj_mgr.add_object_by_template_id(box_template_id)
    # box_obj.translation = [3.2, 0.23, 0.03]
    # box_obj.translation = [3.2, 0.53, 0.03]
    # box_obj.translation = [-0.45, 0.0, -16.4]
    box_obj.translation = place_pos

    transforms_info = {"camera_angle_x": camera_angle_x / (180 / np.pi), "frames": []}

    downward_angles = np.linspace(
        downward_angle_begin, downward_angle_end, render_frame_num
    )

    for idx in tqdm(range(render_frame_num)):
        # Blender (x,y,z) -> glTF(x,z,-y)
        # for glb meshlab (x, y, z) -> (x, z, -y)
        # Twc
        agent_pose = np.eye(4)
        agent_pose[:3, :4] = spheric_pose(
            theta=idx * rotate_per_frame / (180 / np.pi),
            phi=-(downward_angles[idx] * np.pi / 180),
            # phi=0,
            radius=cam_radius,
            height=0,
        )
        # fix pose to y up
        fix_rot = np.eye(4)
        fix_rot[:3, :3] = np.array([1, 0, 0, 0, 0, 1, 0, -1, 0]).reshape(3, 3)
        agent_pose = fix_rot @ agent_pose

        position = np.array([place_pos[0], cam_height, place_pos[2]])
        agent_pose[:3, 3] += position
        # import ipdb; ipdb.set_trace()
        place_agent(sim, agent_pose[:3, :3], agent_pose[:3, 3])
        # print(agent_transform)
        get_obs(sim, show_imgs, save_imgs, idx)

        transforms_info["frames"].append(
            {
                "idx": idx,
                "transform_matrix": agent_pose.tolist(),
                "file_path": f"./full/{idx:05d}",
            }
        )

        write_json(transforms_info, os.path.join(output_path, "transforms_full.json"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-show-images", dest="show_images", action="store_false")
    parser.add_argument("--no-save-images", dest="save_images", action="store_false")
    parser.set_defaults(show_images=True, save_images=True)
    args = parser.parse_args()
    main(show_imgs=args.show_images, save_imgs=args.save_images)
