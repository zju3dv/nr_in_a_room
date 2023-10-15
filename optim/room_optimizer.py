import sys
import os
import ipdb

os.environ["OMP_NUM_THREADS"] = "1"  # noqa
os.environ["MKL_NUM_THREADS"] = "1"  # noqa
sys.path.append(".")  # noqa

import numpy as np
import numba as nb  # mute some warning
import torch
from torch import nn
from collections import defaultdict
import imageio
from tqdm import trange, tqdm
import cv2
import copy
from typing import List, Optional, Any, Dict, Union

from datasets.geo_utils import matrix_to_rotation_6d, rotation_6d_to_matrix
from datasets.ray_utils import (
    get_ray_directions,
    get_ray_directions_equirectangular,
    get_rays,
)
from models.image_attributes import ImageAttributes
from models.multi_rendering import render_rays_multi
from models_neurecon.neus import get_model as get_model_neus
from models_neurecon.neus_multi_rendering import render_rays_multi_neus
from models.nerf_object import NeRF_Object, Embedding
from models import perceptual_model
from utils.util import get_timestamp, read_json, write_point_cloud
from utils import load_ckpt
from utils.bbox_utils import BBoxRayHelper
from optim.misc_utils import (
    adjust_learning_rate,
    detect_keypoints,
    dump_optimization_meta_to_file,
    extract_mesh_from_neus,
    get_instance_mask,
    get_object_meta_info,
    pano_sample_probability,
    seg_mask_to_box_mask,
)
from optim.relation_losses import obj_attach_floor_loss, obj_attach_loss, z_axis_loss
from optim.relation_geo import object_object_attach_loss, object_room_magnetic_loss
from optim.violation_loss import physical_violation_loss
from optim.patch_perceptual import patch_perceptual_loss
from optim.relation_generation import generate_relation_for_all
from optim.viewing_constraint_loss import viewing_constraint_loss


class RoomOptimizer:
    def __init__(
        self,
        scale_factor: float,
        bg_scale_factor: float,
        bg_scene_center: list,
        img_wh: list,
        near: float,
        far: float,
        chunk: int,
        model_ckpt_path_dict: Dict[str, Any],
        config=None,
        scale_factor_dict: Dict[str, Any] = {},
        scene_info_path: str = None,
        scene_info_json_path: str = None,
        model_type="NeuS",
        N_samples: int = 64,
        N_importance: int = 128,
        relation_info: Dict[str, Any] = {},
        output_path: str = None,
        prefix: str = "",
        active_instance_id: list = [46, 4, 9, 102],
        virtual_instance_id: list = [],  # specific for edit (insert virtual to real) mode
        filter_door_and_window: bool = True,
        lr: float = 1e-2,
        N_optim_step: int = 500,
        adjust_lr_per_step: int = 150,
        optim_batch_size: int = 1024,
        use_amp: bool = False,
        extract_obj_bbox_from_neural_model: bool = False,
        ig_data_base_dir: str = "data/ig_dataset_v1.0.1/",
        mask_per_object: bool = False,
        bbox_ray_intersect: bool = True,
        bbox_enlarge: float = 0.1,
        optimize_light_env: bool = True,
        optimize_appearance_code: bool = False,
        use_light_from_image_attr: bool = False,
        use_appearance_from_image_attr: bool = False,
        optimize_option: list = [
            "photometric_loss",
            "perceptual_loss",
            "z_axis_align_loss",
            "object_room_wall_attach",
            "object_room_floor_attach",
            "physical_violation",
            "object_object_attach",
        ],
    ):
        # load config
        self.scene_info_path = scene_info_path
        self.scale_factor = scale_factor
        self.scale_factor_dict = scale_factor_dict
        self.bg_scale_factor = bg_scale_factor
        self.bg_scene_center = np.array(bg_scene_center)
        self.ig_data_base_dir = ig_data_base_dir
        self.mask_per_object = mask_per_object
        self.bbox_ray_intersect = bbox_ray_intersect
        self.bbox_enlarge = bbox_enlarge
        self.virtual_instance_id = virtual_instance_id

        self.img_wh = img_wh
        self.w = img_wh[0]
        self.h = img_wh[1]
        self.near = near
        self.far = far
        self.N_importance = N_importance
        self.N_samples = N_samples
        self.chunk = chunk
        self.lr = lr
        self.N_optim_step = N_optim_step
        self.adjust_lr_per_step = adjust_lr_per_step
        self.optim_batch_size = optim_batch_size
        self.use_amp = use_amp
        self.optimize_light_env = optimize_light_env
        self.optimize_appearance_code = optimize_appearance_code
        self.optimize_option = optimize_option
        self.config = config

        self.use_light_from_image_attr = use_light_from_image_attr
        if self.use_light_from_image_attr:
            print(
                "WARNING: self.use_light_from_image_attr = True, using hard coded light env."
            )
            self.hard_coded_light_id = 0  # just for compatibility
            # self.hard_coded_light_id = 9  # probe_03 in 10 HDR multi_light training

        self.use_appearance_from_image_attr = use_appearance_from_image_attr
        if self.use_appearance_from_image_attr:
            print(
                "WARNING: self.use_appearance_from_image_attr = True, using first frame appearance code."
            )
            self.hard_coded_appearance_frame_id = 0

        self.optimize_exposure = "optimize_exposure" in self.optimize_option

        # laod scene info
        if scene_info_json_path is None:
            scene_info_json_path = os.path.join(scene_info_path, "data.json")
        self.scene_meta = read_json(scene_info_json_path)

        self.active_instance_id = active_instance_id
        if filter_door_and_window:
            self.filter_door_and_window()

        self.relation_info = relation_info

        self.model_type = model_type
        # self.load_model(
        #     model_type, model_ckpt_path_dict["obj"], model_ckpt_path_dict["bg"]
        # )
        self.load_model_from_dict_path(model_type, model_ckpt_path_dict)

        self.reset_optimizable_parameters()

        if extract_obj_bbox_from_neural_model:
            self.extract_bounding_boxes_from_neural_model()

        if self.bbox_ray_intersect:
            self.prepare_bbox_ray_helper()

        self.set_output_path(output_path, prefix)

        print("RoomOptimizer initialize finished.")

    def load_model_from_dict_path(self, model_type, model_ckpt_path_dict):
        assert model_type == "NeuS"
        self.models = {}
        self.image_attrs = {}

        # avoid duplicate loading
        self.models_cache = {}
        self.image_attrs_cache = {}

        print("loading model with instance_id", self.active_instance_id)

        # print(model_ckpt_path_dict)
        for obj_id in self.active_instance_id:
            # identify ckpt_path
            if str(obj_id) in model_ckpt_path_dict:
                ckpt_info = model_ckpt_path_dict[str(obj_id)]
            elif obj_id == 0:
                assert (
                    "bg" in model_ckpt_path_dict or "0" in model_ckpt_path_dict
                ), "model_ckpt_path_dict missing background 'bg' or '0' ckpt"
                ckpt_info = model_ckpt_path_dict.get("bg", model_ckpt_path_dict["0"])
            else:
                print(
                    f"Cannot find specific model for obj_id = {obj_id}, \
                    maybe config file is not compatible with given active_instance_id."
                )
                ckpt_info = model_ckpt_path_dict["obj"]
            # load with cache
            ckpt_path, neus_conf = ckpt_info["path"], ckpt_info["neus_conf"]
            if ckpt_info not in self.models_cache:
                (
                    self.models_cache[ckpt_path],
                    self.image_attrs_cache[ckpt_path],
                ) = self.load_model_neus(ckpt_path, obj_id, neus_conf)
            self.models[f"neus_{obj_id}"] = self.models_cache[ckpt_path]
            self.image_attrs[str(obj_id)] = self.image_attrs_cache[ckpt_path]

    def load_model_nerf(self, ckpt_path):
        # TODO(ybbbbt): fix hard coding
        conf = {
            "N_max_objs": 128,
            "N_obj_embedding": 64,
        }
        nerf_coarse = NeRF_Object(conf)
        nerf_fine = NeRF_Object(conf)
        image_attributes = ImageAttributes(conf)
        load_ckpt(nerf_coarse, ckpt_path, model_name="nerf_coarse")
        load_ckpt(nerf_fine, ckpt_path, model_name="nerf_fine")
        load_ckpt(image_attributes, ckpt_path, model_name="image_attributes")

        nerf_coarse = nerf_coarse.cuda().eval()
        nerf_fine = nerf_fine.cuda().eval()
        image_attributes = image_attributes.cuda().eval()

        models = {
            "coarse": nerf_coarse,
            "fine": nerf_fine,
        }

        embedding_xyz = Embedding(3, 10)
        embedding_dir = Embedding(3, 4)
        embeddings = {
            "xyz": embedding_xyz,
            "dir": embedding_dir,
        }
        return models, embeddings, image_attributes

    def load_model_neus(self, ckpt_path, obj_id, config_path="config/neus.yaml"):
        conf = {
            "model": {
                "N_max_objs": 128,
                "N_obj_embedding": 64,
            },
        }
        if self.optimize_light_env:
            # conf["model"].update({"N_max_lights": 128, "N_light_embedding": 16})
            conf["model"].update({"N_max_lights": 1024, "N_light_embedding": 16})

        if self.optimize_appearance_code and obj_id not in self.virtual_instance_id:
            conf["model"].update(
                {"N_max_appearance_frames": 10000, "N_appearance_embedding": 16}
            )

        neus, render_kwargs_train, render_kwargs_test = get_model_neus(
            config_path=config_path, need_trainer=False, extra_conf=conf
        )
        self.render_kwargs_neus = render_kwargs_test
        image_attributes = ImageAttributes(conf["model"])

        print(ckpt_path)
        load_ckpt(neus, ckpt_path, model_name="neus")
        load_ckpt(image_attributes, ckpt_path, model_name="image_attributes")

        if self.config is not None and (
            str(obj_id) in self.config.get("map_virtual_to_local", {})
        ):
            # image_attributes.embedding_instance
            real_id_in_ckpt = self.config.map_virtual_to_local[str(obj_id)]
            image_attributes.embedding_instance.weight.requires_grad = False
            image_attributes.embedding_instance.weight[
                obj_id
            ] = image_attributes.embedding_instance.weight[real_id_in_ckpt]
            # ipdb.set_trace()

        neus.cuda().eval()
        image_attributes.cuda().eval()
        return neus, image_attributes

    def reset_optimizable_parameters(self):
        self.params = []
        self.relation_info = {}
        if self.optimize_light_env:
            self.initialize_light_code()

        if self.optimize_appearance_code:
            self.initialize_appearance_code()

        if self.optimize_exposure:
            self.initialize_autoexposure()

    def save_optimizable_parameters(self, path):
        all_param_dict = {}
        # all_param_dict["params"] = self.params
        all_param_dict["relation_info"] = self.relation_info
        all_param_dict["object_pose_dict"] = copy.deepcopy(self.object_pose_dict)
        all_param_dict["active_instance_id"] = copy.deepcopy(self.active_instance_id)
        if self.optimize_light_env:
            all_param_dict["light_code"] = copy.deepcopy(self.light_code_dict)
        if self.optimize_appearance_code:
            all_param_dict["appearance_code"] = copy.deepcopy(self.appearance_code_dict)
        if self.optimize_exposure:
            all_param_dict["exposure"] = copy.deepcopy(self.autoexposure_param)
        torch.save(all_param_dict, path)

    def load_optimizable_parameters(self, path):
        all_param_dict = torch.load(path)
        # self.params = all_param_dict["params"]
        self.relation_info = all_param_dict["relation_info"]
        if len(self.virtual_instance_id) == 0:  # not overwrite in edit mode
            self.active_instance_id = all_param_dict["active_instance_id"]

        def to_gpu(code_dict):
            for k, v in code_dict.items():
                if isinstance(v, torch.Tensor):
                    code_dict[k] = v.cuda()
                elif isinstance(v, dict):
                    for k2, v2 in v.items():
                        if isinstance(v2, torch.Tensor):
                            code_dict[k][k2] = v2.cuda()

        if len(self.virtual_instance_id) == 0:  # not modify edit mode pose
            if hasattr(self, "object_pose_dict"):
                self.object_pose_dict.update(all_param_dict["object_pose_dict"])
            else:
                self.object_pose_dict = all_param_dict["object_pose_dict"]
        if self.optimize_light_env:
            self.light_code_dict = all_param_dict["light_code"]
            to_gpu(self.light_code_dict)
        if self.optimize_appearance_code:
            self.appearance_code_dict = all_param_dict["appearance_code"]
            to_gpu(self.appearance_code_dict)
        if self.optimize_exposure and "exposure" in all_param_dict:
            self.autoexposure_param = all_param_dict["exposure"]
            to_gpu(self.autoexposure_param)
        # ipdb.set_trace()

    def interpolate_light_env_from_states(self, path1, path2, interp):
        all_param_dict_1 = torch.load(path1)
        all_param_dict_2 = torch.load(path2)

        # self.params = all_param_dict["params"]
        def to_gpu(code_dict):
            for k, v in code_dict.items():
                if isinstance(v, torch.Tensor):
                    code_dict[k] = v.cuda()
                elif isinstance(v, dict):
                    for k2, v2 in v.items():
                        if isinstance(v2, torch.Tensor):
                            code_dict[k][k2] = v2.cuda()

        if self.optimize_light_env:
            light_code_dict_1 = all_param_dict_1["light_code"]
            light_code_dict_2 = all_param_dict_2["light_code"]
            for k, v in self.light_code_dict.items():
                self.light_code_dict[k] = light_code_dict_1[
                    k
                ] * interp + light_code_dict_2[k] * (1 - interp)
            to_gpu(self.light_code_dict)
        if self.optimize_appearance_code:
            appearance_code_dict_1 = all_param_dict_1["appearance_code"]
            appearance_code_dict_2 = all_param_dict_2["appearance_code"]
            for k, v in self.appearance_code_dict.items():
                self.appearance_code_dict[k] = appearance_code_dict_1[
                    k
                ] * interp + appearance_code_dict_2[k] * (1 - interp)
            to_gpu(self.appearance_code_dict)
        if self.optimize_exposure:
            autoexposure_param_1 = all_param_dict_1["exposure"]
            autoexposure_param_2 = all_param_dict_2["exposure"]
            for k, v in self.autoexposure_param.items():
                self.autoexposure_param[k] = autoexposure_param_1[
                    k
                ] * interp + autoexposure_param_2[k] * (1 - interp)
            to_gpu(self.autoexposure_param)

    def reset_active_instance_id(self, active_instance_id, filter_door_and_window=True):
        self.active_instance_id = active_instance_id
        if filter_door_and_window:
            self.filter_door_and_window()

    def set_output_path(self, output_path: str, prefix: str, with_timestamp=True):
        if output_path is not None:
            if with_timestamp:
                self.output_path = os.path.join(
                    output_path, f"rendered_{get_timestamp()}_{prefix}"
                )
            else:
                self.output_path = os.path.join(output_path, f"{prefix}")
            os.makedirs(self.output_path, exist_ok=True)

    def filter_door_and_window(self):
        print("Filtering door and window objects.")
        filtered_active_instance_id = []
        for obj_id in self.active_instance_id:
            if self.get_type_of_instance(obj_id) not in ["door", "window"]:
                filtered_active_instance_id += [obj_id]
        self.active_instance_id = filtered_active_instance_id

    def initialize_light_code(self):
        self.light_code_dict = {}
        for obj_id in self.active_instance_id:
            # light_code = torch.randn((16)).cuda()
            light_code = torch.zeros((16)).cuda()
            light_code.requires_grad = True
            self.params += [
                {"params": light_code, "lr": self.lr}
            ]  # light code can be optimized with larger lr
            self.light_code_dict[str(obj_id)] = light_code

    def initialize_appearance_code(self):
        self.appearance_code_dict = {}
        for obj_id in self.active_instance_id:
            # appearance_code = torch.randn((16)).cuda()
            appearance_code = torch.zeros((16)).cuda()
            appearance_code.requires_grad = True
            self.params += [
                {"params": appearance_code, "lr": self.lr}
            ]  # light code can be optimized with larger lr
            self.appearance_code_dict[str(obj_id)] = appearance_code

    def initialize_autoexposure(self):
        self.autoexposure_param = {}
        for obj_id in self.active_instance_id:
            # scale and shift
            autoexposure_param = torch.Tensor([1, 1, 1, 0, 0, 0]).cuda()
            autoexposure_param.requires_grad = True
            self.params += [
                {"params": autoexposure_param, "lr": self.lr * 0.1}
            ]  # light code can be optimized with larger lr
            self.autoexposure_param[str(obj_id)] = autoexposure_param

    def get_scale_factor(self, obj_id):
        if obj_id == 0:
            return self.bg_scale_factor
        elif str(obj_id) in self.scale_factor_dict:
            return self.scale_factor_dict[str(obj_id)]
        else:
            return self.scale_factor

    def extract_bounding_boxes_from_neural_model(self):
        print("Extracting object bounding boxes from neural model...")
        assert self.model_type == "NeuS"
        for obj_id in tqdm(self.active_instance_id):
            mesh = extract_mesh_from_neus(
                self.models[f"neus_{obj_id}"],
                self.image_attrs[str(obj_id)],
                obj_id,
            )
            bbox = mesh.get_axis_aligned_bounding_box()
            bound = np.array([bbox.min_bound, bbox.max_bound])
            size = (bound[1] - bound[0]) * self.get_scale_factor(obj_id)
            # update scene_meta
            for idx, obj_info in enumerate(self.scene_meta["objs"]):
                if obj_info["id"] == obj_id:
                    self.scene_meta["objs"][idx]["bdb3d"]["size"] = size.tolist()

    def prepare_bbox_ray_helper(self):
        # bbox ray helper dict
        self.bbox_ray_helper_dict = {}
        for obj_id in self.active_instance_id:
            if obj_id == 0:
                continue
            obj_meta_info = get_object_meta_info(
                self.ig_data_base_dir, self.scene_meta, obj_id
            )
            length = np.array(obj_meta_info["bbox3d"]["size"])
            self.bbox_ray_helper_dict[str(obj_id)] = BBoxRayHelper(np.zeros(3), length)

    def generate_object_rays(
        self, rays_o_obj, rays_d_obj, obj_id, near=None, far=None, select_ind=None
    ):
        """
        Generate object rays given rays_o, rays_d and obj_id
        Input:
            select_ind: only for masked rendering
        """
        if obj_id == 0:  # background
            return self.generate_bg_rays(rays_o_obj, rays_d_obj, near=near, far=far)
        if self.bbox_ray_intersect:
            # for object, rays_o and rays_d should lie in world scale (unscaled)
            bbox_mask, bbox_batch_near, bbox_batch_far = self.bbox_ray_helper_dict[
                str(obj_id)
            ].get_ray_bbox_intersections(
                rays_o_obj,
                rays_d_obj,
                self.get_scale_factor(obj_id),
                # bbox_enlarge=self.bbox_enlarge / self.get_scale_factor(obj_id),
                bbox_enlarge=self.bbox_enlarge,  # in physical world
            )
            # for area which hits bbox, we use bbox hit near far
            # bbox_ray_helper has scale for us, do no need to rescale
            batch_near_obj, batch_far_obj = bbox_batch_near, bbox_batch_far
            rays_o_obj = rays_o_obj / self.get_scale_factor(obj_id)
            # for the invalid part, we use 0 as near far, which assume that (0, 0, 0) is empty
            batch_near_obj[~bbox_mask] = torch.zeros_like(batch_near_obj[~bbox_mask])
            batch_far_obj[~bbox_mask] = torch.zeros_like(batch_far_obj[~bbox_mask])
        else:
            near = self.near if near is None else near
            far = self.far if far is None else far
            batch_near_obj = (
                near
                / self.get_scale_factor(obj_id)
                * torch.ones_like(rays_o_obj[:, :1])
            )
            batch_far_obj = (
                far / self.get_scale_factor(obj_id) * torch.ones_like(rays_d_obj[:, :1])
            )
            rays_o_obj = rays_o_obj / self.get_scale_factor(obj_id)

        if self.mask_per_object:
            # mask out of bound rendering
            obj_mask = torch.from_numpy(self.instance_mask == obj_id).view(-1)
            obj_mask = obj_mask[select_ind]
            batch_near_obj[~obj_mask] = 0
            batch_far_obj[~obj_mask] = 0

        rays_obj = torch.cat(
            [rays_o_obj, rays_d_obj, batch_near_obj, batch_far_obj], 1
        )  # (H*W, 8)
        rays_obj = rays_obj.cuda()
        return rays_obj

    def generate_bg_rays(self, rays_o_bg, rays_d_bg, near=None, far=None):
        near = self.near if near is None else near
        far = self.far if far is None else far
        batch_near_bg = near / self.bg_scale_factor * torch.ones_like(rays_o_bg[:, :1])
        batch_far_bg = far / self.bg_scale_factor * torch.ones_like(rays_d_bg[:, :1])
        rays_o_bg = rays_o_bg / self.bg_scale_factor
        rays_bg = torch.cat(
            [rays_o_bg, rays_d_bg, batch_near_bg, batch_far_bg], 1
        )  # (H*W, 8)
        rays_bg = rays_bg.cuda()
        return rays_bg

    def batched_inference_multi(
        self,
        rays_list,
        obj_id_list,
        to_cpu=True,
        hit_test_only=False,
        need_normal=False,
        use_sphere_tracing=True,
        safe_region_volume_rendering=True,
        refine_edge=False,
        refine_edge_obj_ids=[],
        render_mask=False,
        # use_sphere_tracing=False,
        show_progress=False,
        **kwargs,
    ):
        """Do batched inference on rays using chunk."""
        B = rays_list[0].shape[0]
        results = defaultdict(list)
        for i in tqdm(range(0, B, self.chunk), disable=not show_progress):
            extra_chunk = dict()
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor) and "autoexposure_" not in k:
                    extra_chunk[k] = v[i : i + self.chunk]
                else:
                    extra_chunk[k] = v
            if self.model_type == "NeRF":
                rendered_ray_chunks = render_rays_multi(
                    self.models,
                    self.embeddings,
                    [r[i : i + self.chunk] for r in rays_list],
                    obj_id_list,
                    self.N_samples,
                    use_disp=False,
                    perturb=0.001,
                    # perturb=0.00,
                    noise_std=0,
                    N_importance=self.N_importance,
                    chunk=self.chunk,
                    white_back=True,
                    individual_weight_for_coarse=True,
                    obj_bg_relative_scale=self.bg_scale_factor / self.scale_factor,
                    **extra_chunk,
                )
            elif self.model_type == "NeuS":
                rendered_ray_chunks = render_rays_multi_neus(
                    self,
                    self.models,
                    [r[i : i + self.chunk] for r in rays_list],
                    obj_id_list,
                    noise_std=0,
                    white_back=True,
                    # white_back=False,
                    # obj_bg_relative_scale=self.bg_scale_factor / self.scale_factor,
                    hit_test_only=hit_test_only,
                    need_normal=need_normal,
                    use_sphere_tracing=use_sphere_tracing,
                    safe_region_volume_rendering=safe_region_volume_rendering,
                    refine_edge=refine_edge,
                    refine_edge_obj_ids=refine_edge_obj_ids,
                    render_mask=render_mask,
                    extra_dict=extra_chunk,
                    render_kwargs=self.render_kwargs_neus,
                )

            for k, v in rendered_ray_chunks.items():
                if to_cpu:
                    results[k] += [v.cpu()]
                else:
                    results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results

    def render_full_scene(
        self,
        pose: np.ndarray,
        idx: int,
        h: int,
        w: int,
        write_idx_on_image=True,
        return_raw_image=False,
        render_mask=False,
        refine_edge=False,
        use_sphere_tracing=True,
        safe_region_volume_rendering=False,
        show_progress=False,
        refine_edge_obj_ids=[],
        fovx_deg=0,
    ):
        extra_dict = dict()
        extra_dict["compute_3d_mask"] = False
        extra_dict["is_eval"] = True

        rays_list = []
        object_id_list = []

        if fovx_deg > 0:
            focal = (w / 2) / np.tan((fovx_deg / 2) / (180 / np.pi))
            print("focal =", focal)
            directions = get_ray_directions(h, w, focal).cuda()  # (h, w, 3)
        else:
            directions = get_ray_directions_equirectangular(h, w).cuda()  # (h, w, 3)

        for obj_id in self.active_instance_id:
            # get object location
            # Two: object to world pose
            if obj_id == 0:  # 0 denotes background
                Two = np.eye(4)
                Two[:3, 3] = self.bg_scene_center
            else:  # other objects
                Two = torch.eye(4).cuda()
                Two[:3, :3] = rotation_6d_to_matrix(
                    self.object_pose_dict[str(obj_id)]["rot6d"]
                )
                Two[:3, 3] = self.object_pose_dict[str(obj_id)]["trans"]
                Two = Two.detach().cpu().numpy()
            # pose: Twc
            # we need: Toc
            Twc = np.eye(4)
            Twc[:3, :4] = pose[:3, :4]

            Toc = np.linalg.inv(Two) @ Twc

            Toc = torch.from_numpy(Toc).float().cuda()[:3, :4]
            rays_o, rays_d = get_rays(directions, Toc)

            rays = self.generate_object_rays(rays_o, rays_d, obj_id)

            rays_list += [rays]
            object_id_list += [obj_id]

            # set image_attr for object code
            extra_dict["embedding_inst_{}".format(obj_id)] = self.image_attrs[
                str(obj_id)
            ].embedding_instance(torch.ones_like(rays_o[..., 0]).long().cuda() * obj_id)
            # light code
            if self.optimize_light_env:
                if self.use_light_from_image_attr or obj_id in self.virtual_instance_id:
                    if not hasattr(self, "hard_code_light_id"):
                        self.hard_coded_light_id = 0
                    extra_dict["embedding_light_{}".format(obj_id)] = self.image_attrs[
                        str(obj_id)
                    ].embedding_light(
                        torch.ones_like(rays_o[..., 0]).long().cuda()
                        * self.hard_coded_light_id
                    )
                else:
                    extra_dict["embedding_light_{}".format(obj_id)] = (
                        self.light_code_dict[str(obj_id)]
                        .view(1, -1)
                        .expand(rays_o.shape[0], -1)
                    )
            # appearance code
            if self.optimize_appearance_code and obj_id not in self.virtual_instance_id:
                if self.use_appearance_from_image_attr:
                    extra_dict[
                        "embedding_appearance_{}".format(obj_id)
                    ] = self.image_attrs[str(obj_id)].embedding_appearance(
                        torch.ones_like(rays_o[..., 0]).long().cuda() * 0
                    )
                else:
                    extra_dict["embedding_appearance_{}".format(obj_id)] = (
                        self.appearance_code_dict[str(obj_id)]
                        .view(1, -1)
                        .expand(rays_o.shape[0], -1)
                    )

            # optimize exposure
            if self.optimize_exposure and obj_id not in self.virtual_instance_id:
                extra_dict[f"autoexposure_{obj_id}"] = self.autoexposure_param[
                    str(obj_id)
                ]

        with torch.cuda.amp.autocast(enabled=True):
            with torch.no_grad():
                results = self.batched_inference_multi(
                    rays_list,
                    object_id_list,
                    to_cpu=False,
                    use_sphere_tracing=use_sphere_tracing,
                    # use_sphere_tracing=True,
                    safe_region_volume_rendering=safe_region_volume_rendering,
                    refine_edge=refine_edge,
                    render_mask=render_mask,
                    show_progress=show_progress,
                    **extra_dict,
                )
        img = results[f"rgb_fine"]
        img_pred = np.clip(img.view(h, w, 3).cpu().numpy(), 0, 1)
        img_pred_ = (img_pred * 255).astype(np.uint8)

        if return_raw_image:
            if render_mask:
                img_mask = results[f"rendered_instance_mask"]
                img_mask = (
                    img_mask.view(h, w, 3)[:, :, 0]
                    .cpu()
                    .numpy()
                    .round()
                    .astype(np.uint16)
                )
                return img_pred_, img_mask
            return img_pred_  # raw image in [h, w, 3] np.uint8

        if write_idx_on_image:
            img_pred_ = cv2.putText(
                img_pred_,
                "Iter: {:03d}".format(idx),
                (20, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2,
            )

        imageio.imwrite(
            os.path.join(self.output_path, f"{idx:06d}.multi_obj.png"), img_pred_
        )
        if render_mask:
            img_mask = results[f"rendered_instance_mask"]
            img_mask = (
                img_mask.view(h, w, 3)[:, :, 0].cpu().numpy().round().astype(np.uint16)
            )
            cv2.imwrite(os.path.join(self.output_path, f"{idx:06d}.seg.png"), img_mask)

    def set_initial_object_poses_from_scene_meta(self, add_noise=True):
        self.object_pose_dict = {}

        for obj_id in self.active_instance_id:
            if obj_id == 0:
                continue
            obj_meta_info = get_object_meta_info(
                self.ig_data_base_dir, self.scene_meta, obj_id
            )
            if "gt_T_wo" in obj_meta_info:
                Two = obj_meta_info["gt_T_wo"]
            else:
                print(
                    f"Cannot find object pose for obj_id = {obj_id}, use custom pose with minor offset."
                )
                Two = np.eye(4)
                from scipy.spatial.transform import Rotation as R

                rot_fix = np.array([1, 0, 0, 0, 0, 1, 0, -1, 0]).reshape(3, 3)
                # TODO: update initial pose for real-world scenes
                # if obj_id == 31:
                #     blender_xyz = np.array([-1.44, 1.18, 0.1])
                #     blender_rot = R.from_quat([0.5, -0.5, 0.5, 0.5]).as_matrix()
                # elif obj_id == 32:
                #     blender_xyz = np.array([0.76, 0.54, 0.98])
                #     blender_rot = R.from_quat([0.707107, 0, 0, 0.707107]).as_matrix()
                # elif obj_id == 33:
                #     blender_xyz = np.array([-0.06, 1.01, -0.9])
                #     blender_rot = R.from_quat([0, 0.707107, -0.707107, 0]).as_matrix()
                # elif obj_id == 34:
                #     blender_xyz = np.array([-0.05, 1.14, -0.15])
                #     blender_rot = R.from_quat([0, 0.707107, -0.707107, 0]).as_matrix()
                # elif obj_id == 35:
                #     blender_xyz = np.array([-0.35, 1.1, 0.98])
                #     blender_rot = R.from_quat([0.707107, 0, 0, 0.707107]).as_matrix()

                # Two[:3, :3] = blender_rot @ rot_fix
                # Two[:3, :3] = rot_fix @ blender_rot
                # Two[:3, 3] = rot_fix @ blender_xyz

                # Two[1, 3] += 0.75
                # Two[2, 3] -= 0.7

            # add noise
            if add_noise:
                Two[:3, 3] += 0.1
                from scipy.spatial.transform import Rotation as R

                rot_noise = R.from_euler("z", 20, degrees=True).as_matrix()
                Two[:3, :3] = Two[:3, :3] @ rot_noise
            Two = torch.from_numpy(Two).float().cuda()

            # split parameters
            rot6d = matrix_to_rotation_6d(Two[:3, :3])
            trans = Two[:3, 3]
            rot6d.requires_grad = True
            trans.requires_grad = True

            self.object_pose_dict[str(obj_id)] = {
                "trans": trans,
                "rot6d": rot6d,
            }
            if "fix_object_pose" not in self.optimize_option:
                self.params += [{"params": trans, "lr": self.lr}]
                self.params += [{"params": rot6d, "lr": self.lr}]

    def set_initial_pose_from_prediction(self, pred_json_path):
        print("Initial pose from", pred_json_path)
        self.object_pose_dict = {}
        self.initial_pose_prediction = {}
        pred_info = read_json(pred_json_path)
        for obj_id in self.active_instance_id:
            if obj_id == 0:
                continue
            Two = np.array(pred_info[str(obj_id)]["Two"])
            Two = torch.from_numpy(Two).float().cuda()
            self.initial_pose_prediction[str(obj_id)] = {"Two": Two.clone()}

            # split parameters
            rot6d = matrix_to_rotation_6d(Two[:3, :3])
            trans = Two[:3, 3]

            if not "fix_object_pose" in self.optimize_option:
                rot6d.requires_grad = True
                trans.requires_grad = True

            self.object_pose_dict[str(obj_id)] = {
                "trans": trans,
                "rot6d": rot6d,
            }
            self.params += [{"params": trans, "lr": self.lr}]
            self.params += [{"params": rot6d, "lr": self.lr}]

    def set_initial_pose_as_identity(self):
        print("Initial pose as identity.")
        self.object_pose_dict = {}
        self.initial_pose_prediction = {}
        for obj_id in self.active_instance_id:
            if obj_id == 0:
                continue
            Two = np.eye(4)
            Two = torch.from_numpy(Two).float().cuda()
            self.initial_pose_prediction[str(obj_id)] = {"Two": Two.clone()}

            # split parameters
            rot6d = matrix_to_rotation_6d(Two[:3, :3])
            trans = Two[:3, 3]
            rot6d.requires_grad = True
            trans.requires_grad = True

            self.object_pose_dict[str(obj_id)] = {
                "trans": trans,
                "rot6d": rot6d,
            }
            self.params += [{"params": trans, "lr": self.lr}]
            self.params += [{"params": rot6d, "lr": self.lr}]

    def set_sampling_mask_from_seg(
        self,
        seg_mask=None,
        seg_mask_path=None,
        add_noise_to_seg=0,
        convert_seg_mask_to_box_mask=False,
    ):
        if seg_mask_path is not None:
            print("Read segmentation from gt mask")
            # read mask
            self.instance_mask = get_instance_mask(seg_mask_path, img_wh=self.img_wh)
        elif seg_mask is not None:
            self.instance_mask = seg_mask
        else:
            print("Warning: empty mask")
            self.merged_mask = (
                np.ones((self.img_wh[1], self.img_wh[0])).reshape(-1).astype(bool)
            )
            return

        # merge active object masks
        merged_mask = np.zeros_like(self.instance_mask)
        for i_obj, obj_id in enumerate(self.active_instance_id):
            if obj_id == 0:
                continue  # do not accumulate background obj_id
            instance_mask_obj = self.instance_mask == obj_id
            # use tightly fit bbox instead of segmentation mask
            if convert_seg_mask_to_box_mask:
                instance_mask_obj = seg_mask_to_box_mask(instance_mask_obj)
            merged_mask = np.logical_or(merged_mask, instance_mask_obj)

        # if add noise to gt segmentation
        if add_noise_to_seg != 0:
            is_dilate = add_noise_to_seg > 0
            add_noise_to_seg = abs(add_noise_to_seg)
            kernel = np.ones((add_noise_to_seg, add_noise_to_seg), np.uint8)
            if is_dilate:
                merged_mask = cv2.dilate(
                    merged_mask.astype(np.uint8), kernel, iterations=1
                ).astype(bool)
            else:
                merged_mask = cv2.erode(
                    merged_mask.astype(np.uint8), kernel, iterations=1
                ).astype(bool)
        cv2.imwrite(
            f"{self.output_path}/merged_mask.png", merged_mask.astype(np.uint8) * 255
        )
        self.merged_mask = merged_mask.reshape(-1)

    def get_type_of_instance(self, instance_id):
        for obj_info in self.scene_meta["objs"]:
            if obj_info["id"] == instance_id:
                return obj_info["classname"]
        return "unknown"

    def generate_relation(
        self,
        obj_to_room_distance_th: float = 0.5,
        top_down_dist_th: float = 0.3,
        top_down_xy_close_factor: float = 0.8,
    ):
        """
        Generate relationship : object-wall, object-floor, object-object
        """
        print("Start to generate relation from initial poses and neural models...")
        all_obj_info = {}
        for i, obj_id in enumerate(self.active_instance_id):
            if obj_id == 0:
                continue
            Rwo = rotation_6d_to_matrix(self.object_pose_dict[str(obj_id)]["rot6d"])
            two = self.object_pose_dict[str(obj_id)]["trans"]
            optimized_meta = get_object_meta_info(
                self.ig_data_base_dir, self.scene_meta, obj_id
            )
            optimized_meta.pop("gt_T_wo", None)  # pop gt
            # pass optimized object pose
            optimized_meta["Rwo"] = Rwo
            optimized_meta["two"] = two
            optimized_meta["obj_id"] = obj_id
            all_obj_info[str(obj_id)] = optimized_meta
        with torch.no_grad():
            generate_relation_for_all(
                room_optimizer=self,
                all_obj_info=all_obj_info,
                obj_to_room_distance_th=obj_to_room_distance_th,
                top_down_dist_th=top_down_dist_th,
                top_down_xy_close_factor=top_down_xy_close_factor,
            )
        # print("Relation:\n", self.relation_info)
        for k, v in self.relation_info.items():
            print(k, v)

    def optimize(self, input_rgb: torch.Tensor, pose=None):
        """
        Inputs:
            input_rgb: torch.Tensor [h, w, 3] normalized in 0...1
        """
        if pose is None:
            pose = np.array(self.scene_meta["camera"]["cam3d2world"]).reshape(4, 4)
        # Original poses has rotation in form "right down forward", change to NDC "right up back"
        fix_rot = np.array([1, 0, 0, 0, -1, 0, 0, 0, -1]).reshape(3, 3)
        pose[:3, :3] = pose[:3, :3] @ fix_rot

        # camera to world pose
        Twc = np.eye(4)
        Twc[:3, :4] = pose[:3, :4]
        Twc = torch.from_numpy(Twc).float().cuda()

        if "keypoint_mask" in self.optimize_option:
            # detect keypoint for interest region
            keypoint_mask = detect_keypoints(input_rgb.numpy(), circle_radius=5)
            self.merged_mask = np.logical_and(
                keypoint_mask, self.merged_mask.reshape(keypoint_mask.shape)
            )
            cv2.imwrite(
                f"{self.output_path}/merged_mask_keypoint.png",
                self.merged_mask.astype(np.uint8) * 255,
            )
            self.merged_mask = self.merged_mask.reshape(-1)

        input_rgb = input_rgb.view(-1, 3)  # (H*W, 3) RGB

        directions = get_ray_directions_equirectangular(
            self.h, self.w
        ).cuda()  # (h, w, 3)

        mse_loss = nn.MSELoss(reduction="none")

        assert hasattr(
            self, "params"
        ), "Please set initial pose params before optimization."
        optimizer = torch.optim.Adam(self.params)

        scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        perceptual_net = perceptual_model.VGG16_for_Perceptual().cuda()

        sample_prob = pano_sample_probability(self.h, self.w).reshape(-1)

        t = trange(self.N_optim_step, desc="Opt.", leave=True)
        for i_step in t:
            if "regenerate_relation_during_test" in self.optimize_option:
                if i_step != 0 and i_step % 50 == 0:
                    self.generate_relation()
            if self.adjust_lr_per_step > 0:
                adjust_learning_rate(
                    self.lr,
                    optimizer,
                    i_step,
                    base=0.5,
                    adjust_lr_every=self.adjust_lr_per_step,
                )
            extra_dict = dict()
            rays_list = []
            object_id_list = []
            # sample according to batch size limitation
            select_ind = np.arange(self.merged_mask.shape[0])[self.merged_mask]
            if (
                "perceptual_loss" not in self.optimize_option
            ):  # we only sample some points in this case
                # sample according to pano distribution
                select_sample_prob = sample_prob[self.merged_mask]
                select_sample_prob /= select_sample_prob.sum()
                # assert select_ind.shape[0] > self.optim_batch_size
                sample_size = min(select_ind.shape[0], self.optim_batch_size)
                select_ind = np.random.choice(
                    select_ind,
                    size=sample_size,
                    replace=False,
                    p=select_sample_prob,
                )

            # add some sampling on the background for bg light code
            if self.optimize_light_env:
                bg_sample_ratio = 0.2
                bg_sample_prob = sample_prob[~self.merged_mask]
                bg_sample_prob /= bg_sample_prob.sum()
                bg_sample_ind = np.arange(self.merged_mask.shape[0])[~self.merged_mask]
                # assert bg_sample_ind.shape[0] > self.optim_batch_size
                bg_sample_size = min(
                    bg_sample_ind.shape[0], int(bg_sample_ratio * self.optim_batch_size)
                )
                if bg_sample_size > 0:
                    bg_sample_ind = np.random.choice(
                        bg_sample_ind,
                        size=bg_sample_size,
                        replace=False,
                        p=bg_sample_prob,
                    )
                    select_ind = np.concatenate([select_ind, bg_sample_ind], axis=-1)

            select_ind = np.unique(select_ind)
            if i_step == 0:
                print("Actual optimization rays", select_ind.shape[0])
            select_input_rgb = input_rgb[select_ind].float().cuda()

            loss_dict = {}
            all_obj_info = {}  # prepare for violation loss

            for i, obj_id in enumerate(self.active_instance_id):
                # object to world pose
                if obj_id == 0:
                    Rwo = torch.eye(3).cuda()
                    two = torch.from_numpy(self.bg_scene_center).float().cuda()
                else:
                    Rwo = rotation_6d_to_matrix(
                        self.object_pose_dict[str(obj_id)]["rot6d"]
                    )
                    two = self.object_pose_dict[str(obj_id)]["trans"]

                # camera to object pose
                Toc = torch.eye(4).cuda()
                Toc[:3, :3] = Rwo.T @ Twc[:3, :3]
                Toc[:3, 3] = Rwo.T @ (Twc[:3, 3] - two)

                # generate object rays
                rays_o, rays_d = get_rays(directions, Toc[:3, :4])

                rays_o = rays_o[select_ind]
                rays_d = rays_d[select_ind]

                rays = self.generate_object_rays(rays_o, rays_d, obj_id)
                rays_list += [rays]
                object_id_list += [obj_id]

                # set image_attr for object code
                extra_dict["embedding_inst_{}".format(obj_id)] = self.image_attrs[
                    str(obj_id)
                ].embedding_instance(
                    torch.ones_like(rays_o[..., 0]).long().cuda() * obj_id
                )
                # light code
                if self.optimize_light_env:
                    if self.use_light_from_image_attr:
                        extra_dict[
                            "embedding_light_{}".format(obj_id)
                        ] = self.image_attrs[str(obj_id)].embedding_light(
                            torch.ones_like(rays_o[..., 0]).long().cuda()
                            * self.hard_coded_light_id
                        )
                    else:
                        extra_dict["embedding_light_{}".format(obj_id)] = (
                            self.light_code_dict[str(obj_id)]
                            .view(1, -1)
                            .expand(rays_o.shape[0], -1)
                        )
                #  appearance code
                if self.optimize_appearance_code:
                    if self.use_appearance_from_image_attr:
                        extra_dict[
                            "embedding_appearance_{}".format(obj_id)
                        ] = self.image_attrs[str(obj_id)].embedding_appearance(
                            torch.ones_like(rays_o[..., 0]).long().cuda() * 0
                        )
                    else:
                        extra_dict["embedding_appearance_{}".format(obj_id)] = (
                            self.appearance_code_dict[str(obj_id)]
                            .view(1, -1)
                            .expand(rays_o.shape[0], -1)
                        )
                # autoexposure
                if self.optimize_exposure:
                    extra_dict[f"autoexposure_{obj_id}"] = self.autoexposure_param[
                        str(obj_id)
                    ]

                # we do not need to add relation constraints to bg
                if obj_id == 0:
                    continue

                # enforce optimising on yaw
                if "z_axis_align_loss" in self.optimize_option:
                    loss_dict["z_axis_loss_{}".format(obj_id)] = (
                        z_axis_loss(Rwo, 1.0) * 1e2
                    )

                optimized_meta = get_object_meta_info(
                    self.ig_data_base_dir, self.scene_meta, obj_id
                )
                optimized_meta.pop("gt_T_wo", None)  # pop gt
                # pass optimized object pose
                optimized_meta["Rwo"] = Rwo
                optimized_meta["two"] = two
                optimized_meta["obj_id"] = obj_id
                obj_id_key = str(obj_id)

                if obj_id_key not in self.relation_info:
                    continue

                # get obj_relation from input
                obj_relation = self.relation_info[obj_id_key]
                # supplement obj_type
                obj_type = self.get_type_of_instance(obj_id)
                optimized_meta["obj_type"] = obj_type

                all_obj_info[str(obj_id)] = optimized_meta

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    """attach wall loss"""
                    if (
                        "object_room_wall_attach" in self.optimize_option
                        and obj_relation.get("attach_wall", False)
                    ):
                        kwargs = {
                            "room_optimizer": self,
                            "obj_info": optimized_meta,
                            # "face_direction": torch.Tensor([0, 1, 0]),
                            # "face_direction": obj_relation.get(
                            #     "attach_wall_face_dir", torch.Tensor([0, 1, 0])
                            # ),
                            "face_direction": obj_relation["attach_wall_face_dir"],
                            "ray_grid_size": 10,
                        }
                        # for door object, we slightly stretch the size to ensure successive hit-test
                        if obj_type == "door" or obj_type == "window":
                            kwargs.update(
                                {
                                    "ray_grid_stretch": torch.Tensor([1.2, 1.2, 1]),
                                    "use_bbox_surface_as_in_detect": True,
                                }
                            )
                        loss_dict.update(object_room_magnetic_loss(**kwargs))

                    """attach floor loss"""
                    if (
                        "object_room_floor_attach" in self.optimize_option
                        and obj_relation.get("attach_floor", False)
                    ):
                        #     # TODO(ybbbbt): hard code floor
                        #     loss_dict.update(
                        #         obj_attach_floor_loss(optimized_meta, floor=0.0)
                        #     )
                        kwargs = {
                            "room_optimizer": self,
                            "obj_info": optimized_meta,
                            "face_direction": torch.Tensor([0, 0, -1]),
                            "ray_grid_stretch": torch.Tensor(
                                [0.8, 0.8, 1.0]
                            ),  # avoid too close to wall
                            "use_bbox_surface_as_in_detect": True,
                            "ray_grid_size": 3,
                        }
                        if obj_type == "door":
                            # kwargs["ray_grid_offset"] = torch.Tensor(
                            #     [0, -0.3, 0]
                            # )  # to avoid to close to wall
                            assert (
                                "attach_wall_face_dir" in obj_relation
                            ), f"door {obj_id} relation prediction failed."
                            kwargs["ray_grid_offset"] = (
                                obj_relation["attach_wall_face_dir"] * -0.3
                            )  # to avoid to close to wall
                        loss_dict.update(object_room_magnetic_loss(**kwargs))

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                results = self.batched_inference_multi(
                    rays_list,
                    object_id_list,
                    to_cpu=False,
                    # use_sphere_tracing=True,
                    use_sphere_tracing=False,
                    **extra_dict,
                )
                pred_rgb = results["rgb_fine"]

                if "photometric_loss" in self.optimize_option:
                    loss_dict["mse_loss"] = mse_loss(pred_rgb, select_input_rgb).mean()

                if "visualize_pred" in self.optimize_option:  # dump image for debug
                    # pred_rgb_full = input_rgb.cuda()
                    pred_rgb_full = torch.zeros_like(input_rgb.cuda())
                    pred_rgb_full[select_ind] = pred_rgb

                    imageio.imwrite(
                        f"debug/pred_rgb_full.png",
                        (pred_rgb_full * 255)
                        .view(self.img_wh[1], self.img_wh[0], 3)
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.uint8),
                    )

                if "perceptual_loss" in self.optimize_option:
                    pred_rgb_full = input_rgb.cuda()
                    pred_rgb_full[select_ind] = pred_rgb
                    loss_dict.update(
                        patch_perceptual_loss(
                            perceptual_net,
                            pred_rgb_full,
                            input_rgb,
                            all_obj_info,
                            self.instance_mask,
                            self.img_wh,
                        )
                    )

                """attach bottom to other object loss"""
                if "object_object_attach" in self.optimize_option:
                    for obj_id_str, obj_relation in self.relation_info.items():
                        if obj_relation.get("attach_bottom_to_object", False):
                            kwargs = {
                                "room_optimizer": self,
                                "obj_info_src": all_obj_info[obj_id_str],
                                "obj_info_tgt": all_obj_info[
                                    str(obj_relation["attach_tgt_obj_id"])
                                ],
                                "face_direction": torch.Tensor([0, 0, -1]),
                            }
                            loss_dict.update(object_object_attach_loss(**kwargs))

                # physical violation loss
                if "physical_violation" in self.optimize_option:
                    if (
                        not "physical_violation_delayed_start" in self.optimize_option
                        or i_step >= 100
                    ):
                        loss_dict.update(
                            physical_violation_loss(
                                self,
                                all_obj_info,
                                N_nearest_obj=3,
                                check_background_violation=True,
                                # N_sample_points=1000,
                                N_sample_points=2000,
                                # N_sample_points=300,
                            )
                        )

                if "viewing_constraint" in self.optimize_option:
                    loss_dict.update(viewing_constraint_loss(self, Twc, all_obj_info))

                if "print_loss_dict" in self.optimize_option:
                    for k, v in loss_dict.items():
                        # if "_62" not in k:
                        #     continue
                        print(k, "=", float(v))
                loss = sum(list(loss_dict.values()))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            t.set_description("Loss: %f" % float(loss))
            t.refresh()
            # dump image
            if i_step % 20 == 0:
                self.save_optimizable_parameters(
                    f"{self.output_path}/{i_step:06d}.state.ckpt"
                )
                # self.load_optimizable_parameters(
                #     f"{self.output_path}/{i_step:06d}.state.ckpt"
                # )
                if i_step >= self.N_optim_step - 20:
                    self.render_full_scene(
                        pose=pose,
                        idx=i_step,
                        write_idx_on_image=False,
                        render_mask=True,
                        h=512,
                        w=1280,
                    )
                else:
                    self.render_full_scene(
                        pose=pose,
                        idx=i_step,
                        render_mask=False,
                        h=self.h,
                        w=self.w,
                    )
                dump_optimization_meta_to_file(
                    filepath=f"{self.output_path}/{i_step:06d}.optim.json",
                    obj_pose_dict=self.object_pose_dict,
                )
