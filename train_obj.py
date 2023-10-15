import os

os.environ["OMP_NUM_THREADS"] = "1"  # noqa
os.environ["MKL_NUM_THREADS"] = "1"  # noqa

from utils.util import get_timestamp, make_source_code_snapshot, write_image
import torch
from collections import defaultdict
from omegaconf import OmegaConf

from torch.utils.data import DataLoader
from datasets import dataset_dict

# models
from models.nerf_object import NeRF_Object, Embedding
from models_neurecon import neus
from models.rendering_obj import render_rays_obj
from models.image_attributes import ImageAttributes

# optimizer, scheduler, visualization
from utils import *

# losses
from models.losses import PatchSemanticLoss, get_loss

# metrics
from models.metrics import *

# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TestTubeLogger


class NeRFSystem(LightningModule):
    def __init__(self, config):
        super(NeRFSystem, self).__init__()
        self.config = config

        self.loss = get_loss(config, instance_only=True)
        # self.patch_sem_loss = PatchSemanticLoss()

        self.models_to_train = []

        self.model_type = config.get("model_type", "NeRF_Object")

        # load NeRF object model
        if self.model_type == "NeRF_Object":
            self.embedding_xyz = Embedding(3, 10)

            self.embedding_dir = Embedding(3, 4)
            self.embeddings = {"xyz": self.embedding_xyz, "dir": self.embedding_dir}

            self.nerf_coarse = NeRF_Object(conf=config.model)
            self.models = {"coarse": self.nerf_coarse}
            load_ckpt(self.nerf_coarse, config.ckpt_path, "nerf_coarse")

            if config.model.N_importance > 0:
                self.nerf_fine = NeRF_Object(conf=config.model)
                self.models["fine"] = self.nerf_fine
                load_ckpt(self.nerf_fine, config.ckpt_path, "nerf_fine")
            self.models_to_train += [self.models]

        # load NeuS
        elif self.model_type == "NeuS":
            (
                self.neus,
                self.render_kwargs_train,
                self.render_kwargs_test,
            ) = neus.get_model(
                config_path=config.neus_conf, need_trainer=False, extra_conf=config
            )
            load_ckpt(self.neus, config.ckpt_path, "neus")
            self.models_to_train += [self.neus]

        self.image_attributes = ImageAttributes(conf=config.model)
        load_ckpt(self.image_attributes, config.ckpt_path, "image_attributes")

        self.models_to_train += [self.image_attributes]

        self.save_hyperparameters(config)

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def forward(self, rays, extra=dict()):
        """Do batched inference on rays using chunk."""
        B = rays.shape[0]
        results = defaultdict(list)
        for i in range(0, B, self.config.train.chunk):
            extra_chunk = dict()
            for k, v in extra.items():
                if isinstance(v, torch.Tensor):
                    extra_chunk[k] = v[i : i + self.config.train.chunk]
                else:
                    extra_chunk[k] = v
            if self.model_type == "NeRF_Object":
                rendered_ray_chunks = render_rays_obj(
                    self.models,
                    self.embeddings,
                    rays[i : i + self.config.train.chunk],
                    self.config.model.N_samples,
                    self.config.model.use_disp,
                    self.config.model.perturb,
                    self.config.model.noise_std,
                    self.config.model.N_importance,
                    self.config.train.chunk,  # chunk size is effective in val mode
                    self.train_dataset.white_back
                    if self.training
                    else self.val_dataset.white_back,
                    **extra_chunk,
                )
            elif self.model_type == "NeuS":
                rendered_ray_chunks = neus.render_rays_neus(
                    model=self.neus,
                    rays=rays[i : i + self.config.train.chunk],
                    extra_dict=extra_chunk,
                    render_kwargs=self.render_kwargs_train
                    if self.training
                    else self.render_kwargs_test,
                )

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results

    def setup(self, stage):
        dataset = dataset_dict[self.config.dataset.dataset_name]
        kwargs = {"img_wh": tuple(self.config.dataset.img_wh)}
        if self.config.dataset.dataset_name == "habitat_base":
            kwargs["config"] = self.config.dataset
        elif self.config.dataset.dataset_name.find("single") != -1:
            kwargs["config"] = self.config.dataset
            kwargs["batch_size"] = self.config.train.batch_size
        self.train_dataset = dataset(split="train", **kwargs)
        self.val_dataset = dataset(split="val", **kwargs)

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.config.train, self.models_to_train)
        scheduler = get_scheduler(self.config.train, self.optimizer)
        return [self.optimizer], [scheduler]

    def train_dataloader(self):
        batch_size = (
            None
            if self.config.dataset.dataset_name.find("single") != -1
            else self.config.train.batch_size
        )
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            num_workers=6,
            batch_size=batch_size,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            num_workers=4,
            batch_size=1,  # validate one image (H*W rays) at a time
            pin_memory=True,
        )

    def training_step(self, batch, batch_nb):
        rays, rgbs = batch["rays"], batch["rgbs"]
        rays = rays.squeeze()  # (H*W, 3)
        rgbs = rgbs.squeeze()  # (H*W, 3)

        # get mask for psnr evaluation
        mask = batch["valid_mask"].view(-1, 1).repeat(1, 3)  # (H*W, 3)

        extra_info = dict()
        extra_info["is_eval"] = False
        extra_info["compute_3d_mask"] = True
        extra_info["instance_mask"] = batch["instance_mask"]

        extra_info.update(self.image_attributes(batch))

        results = self(rays, extra_info)

        # for embedding regularization
        batch["embedding_inst"] = extra_info["embedding_inst"]
        batch["embedding_light"] = extra_info["embedding_light"]
        if "embedding_appearance" in extra_info:
            batch["embedding_appearance"] = extra_info["embedding_appearance"]

        loss_sum, loss_dict = self.loss(results, batch, self.current_epoch)

        with torch.no_grad():
            typ = "fine" if "rgb_instance_fine" in results else "coarse"
            psnr_ = psnr(
                results[f"rgb_instance_{typ}"],
                rgbs,
                batch["instance_mask"].view(-1, 1).expand(-1, 3),
            )

        self.log("lr", get_learning_rate(self.optimizer))
        self.log("train/loss", loss_sum)
        for k, v in loss_dict.items():
            self.log(f"train/{k}", v)
        self.log("train/psnr", psnr_, prog_bar=True)

        return loss_sum

    def validation_step(self, batch, batch_nb):
        rays, rgbs = batch["rays"], batch["rgbs"]
        # get mask for psnr evaluation
        if "instance_mask" in batch:
            mask = (
                (batch["valid_mask"] * batch["instance_mask"]).view(-1, 1).repeat(1, 3)
            )  # (H*W, 3)
        else:
            mask = None
        rays = rays.squeeze()  # (H*W, 3)
        rgbs = rgbs.squeeze()  # (H*W, 3)
        extra_info = dict()
        extra_info["is_eval"] = True
        extra_info["compute_3d_mask"] = True
        extra_info.update(self.image_attributes(batch))
        # if 'instance_3d_mask' in batch:
        #     extra_info['instance_3d_mask'] = batch['instance_3d_mask'].squeeze()
        results = self(rays, extra_info)

        # for embedding regularization
        batch["embedding_inst"] = extra_info["embedding_inst"]
        batch["embedding_light"] = extra_info["embedding_light"]
        if "embedding_appearance" in extra_info:
            batch["embedding_appearance"] = extra_info["embedding_appearance"]

        loss_sum, loss_dict = self.loss(results, batch, self.current_epoch)

        for k, v in loss_dict.items():
            self.log(f"val/{k}", v)
        log = {"val_loss": loss_sum}
        log.update(loss_dict)
        typ = "fine" if "rgb_instance_fine" in results else "coarse"

        # if batch_nb == 0:
        if True:
            W, H = self.config.dataset.img_wh
            img_inst = (
                results[f"rgb_instance_{typ}"].view(H, W, 3).permute(2, 0, 1).cpu()
            )  # (3, H, W)
            img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
            if batch["depths"].sum() == 0:
                vis_min, vis_max = None, None
            else:
                vis_min, vis_max = (
                    batch["depths"].min().item(),
                    batch["depths"].max().item(),
                )
                if vis_min < 0.01:
                    vis_min = 0.3
            depth_inst = visualize_depth(
                results[f"depth_instance_{typ}"].view(H, W), vmin=vis_min, vmax=vis_max
            )  # (3, H, W)
            gt_depth = visualize_depth(
                batch["depths"].view(H, W), vmin=vis_min, vmax=vis_max
            )
            opacity = visualize_depth(
                results[f"opacity_instance_{typ}"].unsqueeze(-1).view(H, W),
                vmin=0,
                vmax=1,
            )  # (3, H, W)
            gt_mask = visualize_depth(
                batch["instance_mask"].view(H, W), vmin=0, vmax=1
            )  # (3, H, W)
            stack = torch.stack(
                [img_gt, img_inst, depth_inst, gt_depth, opacity, gt_mask]
            )  # (N, 3, H, W)
            self.logger.experiment.add_images(
                "val/GT_pred_depth", stack, self.global_step + batch_nb
            )
            save_to_file = False
            if save_to_file:
                # (N, 3, H, W) -> (H, N, W, 3) -> (H, N * W, 3)
                image = (
                    stack.permute(2, 0, 3, 1).reshape(H, -1, 3).detach().cpu().numpy()
                )
                image = (image * 255).astype(np.uint8)
                output_dir = os.path.join(self.config.log_path, "dump_vis")
                os.makedirs(output_dir, exist_ok=True)
                obj_id = int(batch["instance_ids"][0][0])
                write_image(
                    os.path.join(
                        output_dir, f"{self.global_step:07d}.obj-{obj_id:03d}.png"
                    ),
                    image,
                )
            # save to file
        psnr_ = psnr(
            results[f"rgb_instance_{typ}"],
            rgbs,
            batch["instance_mask"].view(-1, 1).expand(-1, 3),
        )
        log["val_psnr"] = psnr_

        return log

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        mean_psnr = torch.stack([x["val_psnr"] for x in outputs]).mean()

        self.log("val/loss", mean_loss)
        self.log("val/psnr", mean_psnr, prog_bar=True)
        torch.cuda.empty_cache()


def main(config):
    # exp_name = get_timestamp() + "_" + config.exp_name
    exp_name = config.exp_name
    print(f"Start with exp_name: {exp_name}.")
    config.log_path = f"logs/{exp_name}"

    system = NeRFSystem(config)

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(f"logs/{exp_name}"),
        filename="{epoch:d}",
        monitor="val/psnr",
        mode="max",
        # save_top_k=5,
        save_top_k=-1,
        save_last=True,
        period=1,
    )

    logger = TestTubeLogger(
        save_dir="logs",
        name=exp_name,
        debug=False,
        create_git_tag=False,
        log_graph=False,
    )

    trainer = Trainer(
        max_epochs=config.train.num_epochs,
        callbacks=[checkpoint_callback],
        resume_from_checkpoint=config.ckpt_path,
        logger=logger,
        weights_summary=None,
        progress_bar_refresh_rate=1,
        gpus=config.train.num_gpus,
        accelerator="ddp" if config.train.num_gpus > 1 else None,
        num_sanity_val_steps=1,
        benchmark=True,
        profiler="simple" if config.train.num_gpus == 1 else None,
        val_check_interval=0.25,
        limit_train_batches=config.train.limit_train_batches,
        # precision=16
        precision=32,
    )

    backup_path = f"logs/{exp_name}"
    make_source_code_snapshot(backup_path)
    OmegaConf.save(
        config=config, f=os.path.join(backup_path, "run_config_snapshot.yaml")
    )
    trainer.fit(system)


if __name__ == "__main__":
    conf_cli = OmegaConf.from_cli()
    conf_user_file = OmegaConf.load(conf_cli.config)
    if "parent_config" in conf_user_file:
        conf_parent_file = OmegaConf.load(conf_user_file.parent_config)
    else:
        conf_parent_file = OmegaConf.create()
    conf_default = OmegaConf.load("config/default_conf.yml")
    # merge conf with the priority
    conf_merged = OmegaConf.merge(
        conf_default, conf_parent_file, conf_user_file, conf_cli
    )

    print("-" * 40)
    print(OmegaConf.to_yaml(conf_merged))
    print("-" * 40)

    main(config=conf_merged)
