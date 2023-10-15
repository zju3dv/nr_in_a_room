import torch
from torch import nn
import torch.nn.functional as F
import json
from einops import rearrange, reduce, repeat
from models.perceptual_model import CLIP_for_Perceptual


class OpacityLoss(nn.Module):
    def __init__(self, coef=1):
        super().__init__()
        self.coef = coef
        # self.loss = nn.BCELoss(reduction='mean')
        # self.loss = nn.MSELoss(reduction='mean')
        self.loss = nn.MSELoss(reduction="none")

    def forward(self, inputs, batch):
        valid_mask = batch["valid_mask"].view(-1)
        if valid_mask.sum() == 0:  # skip when mask is empty
            return None
        instance_mask = batch["instance_mask"].view(-1)[valid_mask]
        instance_mask_weight = batch["instance_mask_weight"].view(-1)[valid_mask]
        # TODO(ybbbbt): do we need to learn depth and color specifically for mask area?
        loss = 0
        if "opacity_instance_coarse" in inputs:
            loss += (
                self.loss(
                    torch.clamp(inputs["opacity_instance_coarse"][valid_mask], 0, 1),
                    instance_mask.float(),
                )
                * instance_mask_weight
            ).mean()
        if "rgb_instance_fine" in inputs:
            loss += (
                self.loss(
                    torch.clamp(inputs["opacity_instance_fine"][valid_mask], 0, 1),
                    instance_mask.float(),
                )
                * instance_mask_weight
            ).mean()
        # loss = self.loss(torch.clamp(inputs['opacity_fine'][valid_mask], 0, 1), instance_mask.float())
        return self.coef * loss


class DepthLoss(nn.Module):
    def __init__(self, coef=1, instance_only=False):
        super().__init__()
        self.coef = coef
        self.loss = nn.MSELoss(reduction="none")
        self.instance_only = instance_only

    def forward(self, inputs, batch):
        targets = batch["depths"].view(-1)
        if (targets > 0).sum() == 0:
            return None
        # mask = (batch['valid_mask'] * batch['instance_mask']
        #         * (targets > 0)).view(-1)  # (H*W)
        mask = (batch["valid_mask"] * (targets > 0)).view(-1)  # (H*W)
        if self.instance_only:
            mask = mask * batch["instance_mask"].view(-1)
            if mask.sum() == 0:  # skip when instance mask is empty
                return None
            # opa_coarse = inputs['opacity_instance_coarse'].view(-1)[mask]
            instance_mask_weight = batch["instance_mask_weight"].view(-1)[mask]
            # loss = (self.loss(inputs['depth_instance_coarse'][mask], targets[mask]) * opa_coarse).mean()
            loss = 0
            if "depth_instance_coarse" in inputs:
                loss += (
                    self.loss(inputs["depth_instance_coarse"][mask], targets[mask])
                    * instance_mask_weight
                ).mean()
            if "depth_instance_fine" in inputs:
                opa_fine = inputs["opacity_instance_fine"].view(-1)[mask]
                # loss += (self.loss(inputs['depth_instance_fine'][mask], targets[mask]) * opa_fine).mean()
                loss += (
                    self.loss(inputs["depth_instance_fine"][mask], targets[mask])
                    * instance_mask_weight
                ).mean()
        else:
            loss = self.loss(inputs["depth_coarse"][mask], targets[mask]).mean()
            if "rgb_instance_fine" in inputs:
                loss += self.loss(inputs["depth_fine"][mask], targets[mask]).mean()
        return self.coef * loss


class ColorLoss(nn.Module):
    def __init__(self, coef=1, instance_only=False):
        super().__init__()
        self.coef = coef
        self.loss = nn.MSELoss(reduction="none")
        self.instance_only = instance_only

    def forward(self, inputs, batch):
        targets = batch["rgbs"].view(-1, 3)
        mask = batch["valid_mask"].view(-1, 1).repeat(1, 3)  # (H*W, 3)
        if self.instance_only:
            mask = mask * batch["instance_mask"].view(-1, 1).repeat(1, 3)
            if mask.sum() == 0:  # skip when instance mask is empty
                return None
            # opa_coarse = inputs['opacity_instance_coarse'].view(-1, 1).repeat(1, 3)[mask]
            instance_mask_weight = (
                batch["instance_mask_weight"].view(-1, 1).repeat(1, 3)[mask]
            )
            loss = 0
            if "rgb_instance_coarse" in inputs:
                loss += (
                    self.loss(inputs["rgb_instance_coarse"][mask], targets[mask])
                    * instance_mask_weight
                ).mean()
            # loss = (self.loss(inputs['rgb_instance_coarse'][mask], targets[mask]) * opa_coarse).mean()
            if "rgb_instance_fine" in inputs:
                opa_fine = (
                    inputs["opacity_instance_fine"].view(-1, 1).repeat(1, 3)[mask]
                )
                # loss += (self.loss(inputs['rgb_instance_fine'][mask], targets[mask]) * opa_fine).mean()
                loss += (
                    self.loss(inputs["rgb_instance_fine"][mask], targets[mask])
                    * instance_mask_weight
                ).mean()
        else:
            loss = self.loss(inputs["rgb_coarse"][mask], targets[mask]).mean()
            if "rgb_fine" in inputs:
                loss += self.loss(inputs["rgb_fine"][mask], targets[mask]).mean()

        return self.coef * loss


class EikonalLoss(nn.Module):
    def __init__(self, coef=1):
        super().__init__()
        self.coef = coef

    def forward(self, inputs):
        nablas = inputs["implicit_nablas"]
        nablas_norm = torch.norm(nablas, dim=-1)
        loss = F.mse_loss(nablas_norm, torch.ones_like(nablas_norm))
        return self.coef * loss


class TotalLoss(nn.Module):
    def __init__(self, conf, instance_only=False):
        super().__init__()
        self.conf = conf
        self.instance_only = instance_only
        if not instance_only:
            self.color_loss = ColorLoss(self.conf["color_loss_weight"])
            self.depth_loss = DepthLoss(self.conf["depth_loss_weight"])
        self.opacity_loss = OpacityLoss(self.conf["opacity_loss_weight"])
        self.instance_color_loss = ColorLoss(
            self.conf["instance_color_loss_weight"], True
        )
        self.instance_depth_loss = DepthLoss(
            self.conf["instance_depth_loss_weight"], True
        )
        self.latent_loss_weight = self.conf.get("latent_loss_weight", 0)
        self.eiknoal_loss = EikonalLoss(self.conf.get("eikonal_loss_weight", 0))

    def forward(self, inputs, batch, epoch=-1):
        loss_dict = dict()
        if not self.instance_only:
            loss_dict["color_loss"] = self.color_loss(inputs, batch)
            loss_dict["depth_loss"] = self.depth_loss(inputs, batch)
        if epoch >= 0:
            # TODO(ybbbbt): we need opacity loss for mask training, or the background may be messy
            loss_dict["opacity_loss"] = self.opacity_loss(inputs, batch)
            loss_dict["instance_color_loss"] = self.instance_color_loss(inputs, batch)
            loss_dict["instance_depth_loss"] = self.instance_depth_loss(inputs, batch)

        if self.eiknoal_loss.coef > 0 and "implicit_nablas" in inputs:
            loss_dict["eikonal_loss"] = self.eiknoal_loss(inputs)

        # latent loss
        if "embedding_inst" in batch and self.latent_loss_weight > 0:
            latent_loss = batch["embedding_inst"].norm(dim=1).mean()
            loss_dict["latent_loss"] = latent_loss * self.latent_loss_weight

        # remove unused loss
        loss_dict = {k: v for k, v in loss_dict.items() if v != None}

        loss_sum = sum(list(loss_dict.values()))
        # check depth loss without supervision
        # loss_sum = loss_dict['color_loss']

        # recover loss to orig scale for comparison
        for k, v in loss_dict.items():
            if f"{k}_weight" in self.conf:
                loss_dict[k] /= self.conf[f"{k}_weight"]

        return loss_sum, loss_dict


class PatchSemanticLoss(nn.Module):
    def __init__(self, coef=1):
        super().__init__()
        self.coef = coef
        self.loss = nn.MSELoss(reduction="none")
        self.perceptual_net = CLIP_for_Perceptual()

    def forward(self, inputs, batch):
        patch_h, patch_w = batch["patch_hw"]
        patch_rgb = inputs["rgb_instance_fine"]
        instance_emb = self.perceptual_net.compute_img_embedding(
            patch_rgb.permute(1, 0).view(1, 3, patch_h, patch_w)
        ).squeeze(0)
        loss = self.loss(instance_emb, batch["ref_patch_embedding"]).mean()

        return self.coef * loss


def get_loss(config, instance_only=False):
    loss_conf_dict = dict(config.loss)
    loss_conf_dict["img_wh"] = config.dataset.img_wh

    return TotalLoss(loss_conf_dict, instance_only)
