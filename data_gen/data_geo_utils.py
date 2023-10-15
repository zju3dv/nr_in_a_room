import numpy as np


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


def anorm(x, axis=None, keepdims=False):
    """Compute L2 norms along specified axes."""
    return np.sqrt((x * x).sum(axis=axis, keepdims=keepdims))


def normalize(v, axis=None, eps=1e-10):
    """L2 Normalize along specified axes."""
    return v / max(anorm(v, axis=axis, keepdims=True), eps)


def lookat(eye, target=[0, 0, 0], up=[0, 1, 0]):
    """Generate LookAt modelview matrix."""
    """
        world2cam3d = lookat(self.camera['pos'], self.camera['target'], self.camera['up'])
        world2cam3d = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32) @ world2cam3d
        self.camera['world2cam3d'] = world2cam3d
        self.camera['cam3d2world'] = np.linalg.inv(world2cam3d)
    """
    eye = np.float32(eye)
    forward = normalize(target - eye)
    side = normalize(np.cross(forward, up))
    up = np.cross(side, forward)
    M = np.eye(4, dtype=np.float32)
    R = M[:3, :3]
    R[:] = [side, up, -forward]
    M[:3, 3] = -R.dot(eye)
    return M


def create_sphere_lookat_poses(
    radius: float, n_poses: int, n_circles: float, up_dir="y", phi_begin=20, phi_end=90
):
    deg2rad = np.pi / 180
    # y up
    phi_list = np.linspace(phi_begin * deg2rad, phi_end * deg2rad, n_poses)
    theta_list = np.linspace(0, 360 * deg2rad * n_circles, n_poses)
    poses = []
    eyes = []
    for phi, theta in zip(phi_list, theta_list):
        if up_dir == "y":
            eye = np.array(
                [
                    radius * np.sin(phi) * np.sin(theta),
                    radius * np.cos(phi),
                    radius * np.sin(phi) * np.cos(theta),
                ]
            )
            pose = lookat(eye, target=[0, 0, 0], up=[0, 1, 0])
        elif up_dir == "z":
            eye = np.array(
                [
                    radius * np.sin(phi) * np.sin(theta),
                    radius * np.sin(phi) * np.cos(theta),
                    radius * np.cos(phi),
                ]
            )
            pose = lookat(eye, target=[0, 0, 0], up=[0, 0, 1])
        pose = np.linalg.inv(pose)
        poses += [pose]
        eyes += [eye]
    return poses, eyes
