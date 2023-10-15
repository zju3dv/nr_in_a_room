import torch
from torch import nn


class ImageAttributes(nn.Module):
    """
    Store autoexposure, image subtle warping...
    """

    def __init__(self, conf):
        super(ImageAttributes, self).__init__()
        self.conf = conf
        # TODO(ybbbbt): no more than 128 objects
        self.embedding_instance = torch.nn.Embedding(
            conf["N_max_objs"], conf["N_obj_embedding"]
        )
        if "N_max_lights" in conf:
            self.embedding_light = torch.nn.Embedding(
                conf["N_max_lights"], conf["N_light_embedding"]
            )
        if "N_max_appearance_frames" in conf:
            self.embedding_appearance = torch.nn.Embedding(
                conf["N_max_appearance_frames"], conf["N_appearance_embedding"]
            )

    def forward(self, inputs):
        ret_dict = dict()
        if "instance_ids" in inputs:
            ret_dict["embedding_inst"] = self.embedding_instance(
                inputs["instance_ids"].squeeze()
            )
            if hasattr(self, "embedding_light"):
                ret_dict["embedding_light"] = self.embedding_light(
                    inputs["light_env_ids"].squeeze()
                )
            if hasattr(self, "embedding_appearance"):
                ret_dict["embedding_appearance"] = self.embedding_appearance(
                    inputs["frame_ids"].squeeze()
                )

        return ret_dict
