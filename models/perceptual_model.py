# adopt from improved-nerfmm

from typing import Union

import torch
import torch.nn as nn
from torchvision.transforms import Normalize
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms.transforms import Compose
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize


class VGG16_for_Perceptual(nn.Module):
    def __init__(self, requires_grad=False, n_layers=[2, 4, 14, 21]):
        super(VGG16_for_Perceptual, self).__init__()
        from torchvision import models

        vgg_pretrained_features = models.vgg16(
            pretrained=True
        ).features  # TODO: check requires_grad

        self.slice0 = nn.Sequential()
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()

        for x in range(n_layers[0]):  # relu1_1
            self.slice0.add_module(str(x), vgg_pretrained_features[x])
        for x in range(n_layers[0], n_layers[1]):  # relu1_2
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(n_layers[1], n_layers[2]):  # relu3_2
            self.slice2.add_module(str(x), vgg_pretrained_features[x])

        for x in range(n_layers[2], n_layers[3]):  # relu4_2
            self.slice3.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        # TODO: normalize

        h0 = self.slice0(x)
        h1 = self.slice1(h0)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)

        return h0, h1, h2, h3

    def perceptual_loss(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        low_level: bool = True,
    ):
        def mse_loss(source, target):
            return torch.mean((source - target) ** 2)

        perceptual_loss = 0

        if low_level:
            pred_0 = self.slice0(pred)
            gt_0 = self.slice0(gt)
            perceptual_loss += mse_loss(pred_0, gt_0)
        else:
            pred_0, pred_1, pred_2, pred_3 = self.forward(pred)
            gt_0, gt_1, gt_2, gt_3 = self.forward(gt)

            perceptual_loss += mse_loss(pred_0, gt_0)
            perceptual_loss += mse_loss(pred_1, gt_1)
            perceptual_loss += mse_loss(pred_2, gt_2)
            perceptual_loss += mse_loss(pred_3, gt_3)

        return perceptual_loss


class CLIP_for_Perceptual(nn.Module):
    def __init__(self, requires_grad=False, model_name="ViT-B/32"):
        super().__init__()
        import clip

        self.model, transform_ = clip.load(model_name)
        self.model
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

        # self.norm = None
        # for trans in transform_.transforms:
        #     if isinstance(trans, Normalize):
        #         self.norm = trans

        # self.transform = transform_
        # self.transform = Compose([
        #     Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        #     Normalize((0.48145466, 0.4578275, 0.40821073),
        #               (0.26862954, 0.26130258, 0.27577711)),
        # ])

    def perceptual_loss(self, pred: torch.Tensor, gt: torch.Tensor, **kwargs):
        def mse_loss(source, target):
            return torch.mean((source - target) ** 2)

        def sim_loss(source, target):
            source /= source.norm(dim=-1, keepdim=True)
            target /= target.norm(dim=-1, keepdim=True)
            # import ipdb; ipdb.set_trace()
            return 1 - source @ target.T

        # if self.norm is not None:
        #     pred = self.norm(pred)
        #     gt = self.norm(gt)
        # if hasattr(self, 'transform'):
        #     pred = self.transform(pred)
        #     gt = self.transform(gt)
        f_pred = self.model.encode_image(pred)
        f_gt = self.model.encode_image(gt)
        return mse_loss(f_pred, f_gt)
        # return sim_loss(f_pred, f_gt)

    def compute_img_embedding(self, img: torch.Tensor):
        """
        img:   [B, 3, H, W]
        """
        norm = Normalize(
            (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        )
        img = norm(nn.Upsample((244, 244), mode="bilinear", align_corners=True)(img))
        return self.model.encode_image(img)


def get_perceptual_loss(
    perceptual_net: Union[VGG16_for_Perceptual, CLIP_for_Perceptual],
    pred: torch.Tensor,
    gt: torch.Tensor,
    low_level: bool = True,
):
    """
    perceptual loss is suitable for whole images, not sampled rays.
        pred:   [B, 3, H, W]
        gt:     [B, 3, H, W]
    """
    assert pred.shape == gt.shape
    if pred.shape[2:4] != torch.Size((244, 244)):
        norm = Normalize(
            (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        )
        # pred = norm(nn.Upsample((244, 244), mode="bilinear", align_corners=True)(pred))
        # gt = norm(nn.Upsample((244, 244), mode="bilinear", align_corners=True)(gt))
        pred = norm(pred)
        gt = norm(gt)

    return perceptual_net.perceptual_loss(pred, gt, low_level=low_level)
