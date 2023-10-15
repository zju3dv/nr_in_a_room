import torch


def z_axis_loss(rotmat, weight):
    z_vec = torch.Tensor([0, 0, 1]).float().cuda()
    loss_z_axis = ((rotmat @ z_vec).dot(z_vec) - 1).abs()
    return loss_z_axis * weight


def obj_attach_loss(obj_info, layout_info):
    left_bottom = torch.Tensor([-1, 1, -1]).float().cuda()
    right_bottom = torch.Tensor([1, 1, -1]).float().cuda()
    Rwo = obj_info["Rwo"]
    two = obj_info["two"]
    center = two
    rotation = Rwo
    length = torch.Tensor(obj_info["bbox3d"]["size"]).float().cuda() * 0.5
    left_bottom = center + rotation @ (length * left_bottom)
    right_bottom = center + rotation @ (length * right_bottom)

    left_bottom_2d = left_bottom[:2]
    right_bottom_2d = right_bottom[:2]

    obj_layout = layout_info[str(obj_info["obj_id"])]
    wall_pts = torch.Tensor(obj_layout["wall"]).float().cuda()
    wall_pt1 = wall_pts[0, :2]
    wall_pt2 = wall_pts[1, :2]
    dA_norm = (wall_pt2 - wall_pt1).norm()
    # point to line distance in 2D space
    dist_1 = (
        (wall_pt2[0] - wall_pt1[0]) * (wall_pt1[1] - left_bottom_2d[1])
        - (wall_pt1[0] - left_bottom_2d[0]) * (wall_pt2[1] - wall_pt1[1])
    ).abs() / dA_norm
    dist_2 = (
        (wall_pt2[0] - wall_pt1[0]) * (wall_pt1[1] - right_bottom_2d[1])
        - (wall_pt1[0] - right_bottom_2d[0]) * (wall_pt2[1] - wall_pt1[1])
    ).abs() / dA_norm
    dist_to_floor = torch.abs(left_bottom[2] - obj_layout["floor"])

    return dist_1 + dist_2 + dist_to_floor


def obj_attach_floor_loss(obj_info, floor=0.0):
    obj_id = obj_info["obj_id"]
    left_bottom = torch.Tensor([-1, 1, -1]).float().cuda()
    # right_bottom = torch.Tensor([1, 1, -1]).float().cuda()
    Rwo = obj_info["Rwo"]
    two = obj_info["two"]
    center = two
    rotation = Rwo
    length = torch.Tensor(obj_info["bbox3d"]["size"]).float().cuda() * 0.5
    left_bottom = center + rotation @ (length * left_bottom)
    # right_bottom = center + rotation @ (length * right_bottom)
    # point to line distance in 2D space
    dist_to_floor = torch.abs(left_bottom[2] - floor)

    return {
        f"obj_attach_floor_loss_{obj_id}": dist_to_floor,
    }
