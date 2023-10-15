import os

import cv2
import igibson
import numpy as np
import py360convert

from data_gen.ig_data_config import IG59CLASSES

hdr_texture = os.path.join(
    igibson.ig_dataset_path, "scenes", "background", "probe_02.hdr"
)
hdr_texture2 = os.path.join(
    igibson.ig_dataset_path, "scenes", "background", "probe_03.hdr"
)
background_texture = os.path.join(
    igibson.ig_dataset_path, "scenes", "background", "urban_street_01.jpg"
)


def render_camera(
    renderer,
    camera,
    render_types: (list, str),
    perspective=None,
    obj_groups=None,
    objects_by_id=None,
):
    if isinstance(render_types, str):
        render_types = [render_types]
    # map render types
    render_type_mapping = {"sem": "seg", "depth": "3d"}
    igibson_types = set()
    for save_type in render_types:
        igibson_types.add(render_type_mapping.get(save_type, save_type))

    # render
    if perspective is None:
        perspective = "K" in camera
    if perspective:
        renderer.set_fov(camera["vertical_fov"])
        renderer.set_camera(camera["pos"], camera["target"], camera["up"])
        render_results = renderer.render(modes=igibson_types)
        render_results = {t: r for t, r in zip(igibson_types, render_results)}
        for t, r in render_results.items():
            interpolation = cv2.INTER_LINEAR if t == "rgb" else cv2.INTER_NEAREST
            render_results[t] = cv2.resize(
                r, (camera["width"], camera["height"]), interpolation=interpolation
            )
    else:
        renderer.set_fov(90)
        render_results = render_pano(renderer, camera, igibson_types)

    render_results = {
        t: render_results[render_type_mapping.get(t, t)] for t in render_types
    }

    # convert igibson format
    for render_type, im in render_results.items():
        im = im[:, :, :3].copy()
        if render_type == "seg":
            im = im[:, :, 0]
            im = (im * 255).astype(np.uint8)
            ids = np.unique(im)
            if obj_groups:
                # merge sub objects and super object (for example pillows)
                # into the main sub object (for example bed)
                for main_object, sub_objs in obj_groups.items():
                    for i_subobj in sub_objs:
                        if i_subobj in ids:
                            im[im == i_subobj] = main_object
        elif render_type == "sem":
            seg = im[:, :, 0]
            seg = (seg * 255).astype(np.uint8)
            instances = np.unique(seg)
            im = seg.copy()
            for instance in instances:
                category = objects_by_id[instance].category
                class_id = IG59CLASSES.index(category)
                im[seg == instance] = class_id
        elif render_type == "depth":
            if "K" in camera:
                im = -im[:, :, 2]
            else:
                im = np.linalg.norm(im, axis=-1)
        else:
            im = (im * 255).astype(np.uint8)
        render_results[render_type] = im

    return render_results


def render_pano(renderer, camera, igibson_types):
    forward_v = camera["target"] - camera["pos"]
    left_v = np.array([-forward_v[1], forward_v[0], 0])
    up_v = np.array(camera["up"])
    rot_mat = np.stack([forward_v, left_v, up_v]).T
    cubemaps = {render_type: {} for render_type in igibson_types}
    for direction, up, name in [
        [[-1, 0, 0], [0, 0, 1], "B"],
        [[0, 0, -1], [1, 0, 0], "D"],
        [[0, 1, 0], [0, 0, 1], "L"],
        [[0, -1, 0], [0, 0, 1], "R"],
        [[0, 0, 1], [-1, 0, 0], "U"],
        [[1, 0, 0], [0, 0, 1], "F"],
    ]:
        direction = np.matmul(rot_mat, np.array(direction))
        up = np.matmul(rot_mat, np.array(up))
        renderer.set_camera(camera["pos"], camera["pos"] + direction, up)
        frame = renderer.render(modes=igibson_types)
        for i_render, render_type in enumerate(igibson_types):
            cubemaps[render_type][name] = frame[i_render]
    render_results = {}
    for render_type in igibson_types:
        cubemaps[render_type]["R"] = np.flip(cubemaps[render_type]["R"], 1)
        cubemaps[render_type]["B"] = np.flip(cubemaps[render_type]["B"], 1)
        cubemaps[render_type]["U"] = np.flip(cubemaps[render_type]["U"], 0)
        pano = py360convert.c2e(
            cubemaps[render_type],
            camera["height"],
            camera["width"],
            mode="bilinear" if render_type == "rgb" else "nearest",
            cube_format="dict",
        )
        pano = pano.astype(np.float32)
        render_results[render_type] = pano
    return render_results


def is_obj_valid(obj_dict, min_contour_area=20, min_contour_len=30, min_bdb2d_width=10):
    # check contour length and area
    contour = obj_dict["contour"]
    if contour["area"] < min_contour_area:
        return False
    if len(contour["x"]) < min_contour_len:
        return False

    # check bdb2d width
    bdb2d = obj_dict["bdb2d"]
    if (
        bdb2d["x2"] - bdb2d["x1"] < min_bdb2d_width
        or bdb2d["y2"] - bdb2d["y1"] < min_bdb2d_width
    ):
        return False

    return True
