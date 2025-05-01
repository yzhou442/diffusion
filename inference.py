#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Inference script for DDPM / DDIM models with optional classifier-free guidance,
latent-space diffusion, and FID / IS evaluation.
"""
import os
import sys
import argparse
import numpy as np
import ruamel.yaml as yaml
import torch
import wandb
import logging
from logging import getLogger as get_logger
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F

from torchvision.utils import make_grid, save_image
from torchvision import transforms

# ----------------------------------------------
# 3rd-party / local imports
# ----------------------------------------------
from models import UNet, VAE, ClassEmbedder                # user-defined
from schedulers import DDPMScheduler, DDIMScheduler        # user-defined
from pipelines import DDPMPipeline                         # user-defined
from utils import seed_everything, load_checkpoint         # user-defined
from train import parse_args                               # reuse training args

logger = get_logger(__name__)


# -------------------------------------------------
# Helper: load real validation images for FID/IS
# -------------------------------------------------
def load_reference_images(path: str, image_size: int, limit: int = None):
    """Load reference images into memory as torch.Tensor in [0,1]."""
    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=Image.BICUBIC),
        transforms.ToTensor()
    ])
    files = sorted([
        os.path.join(path, f) for f in os.listdir(path)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    ])
    if limit:
        files = files[:limit]

    imgs = [tfm(Image.open(fp).convert("RGB")) for fp in files]
    return torch.stack(imgs)


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    # ------------------------------------------------------------------
    # 1. Parse arguments – we reuse train.py::parse_args for consistency
    # ------------------------------------------------------------------
    args = parse_args()
    if not hasattr(args, "val_dir"):
        args.val_dir = "./data/val"          # real images directory
    if not hasattr(args, "ckpt"):
        args.ckpt = "./checkpoints/model.pth"

    # ------------------------------------------------------------------
    # 2. Reproducibility
    # ------------------------------------------------------------------
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device=device).manual_seed(args.seed)

    # ------------------------------------------------------------------
    # 3. Logging
    # ------------------------------------------------------------------
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # ------------------------------------------------------------------
    # 4. Build networks
    # ------------------------------------------------------------------
    logger.info("Building UNet …")
    unet = UNet(
        input_size=args.unet_in_size,
        input_ch=args.unet_in_ch,
        T=args.num_train_timesteps,
        ch=args.unet_ch,
        ch_mult=args.unet_ch_mult,
        attn=args.unet_attn,
        num_res_blocks=args.unet_num_res_blocks,
        dropout=args.unet_dropout,
        conditional=args.use_cfg,
        c_dim=args.num_classes if args.use_cfg else None,   # class-conditional dim
    )
    logger.info(f"UNet params: {sum(p.numel() for p in unet.parameters())/1e6:.2f} M")

    # (Optional) latent-diffusion VAE
    vae = None
    if args.latent_ddpm:
        logger.info("Loading VAE …")
        vae = VAE()
        vae.init_from_ckpt(args.vae_ckpt)          # e.g. "pretrained/vae.ckpt"
        vae.eval()

    # Classifier-free guidance: learnable class embeddings
    class_embedder = None
    if args.use_cfg:
        logger.info("Building class embedder …")
        # dim should match UNet channel multiplier’s last value
        embed_dim = args.unet_ch * args.unet_ch_mult[-1]
        class_embedder = ClassEmbedder(num_classes=args.num_classes,
                                       embed_dim=embed_dim)

    # ------------------------------------------------------------------
    # 5. Scheduler (DDPM or DDIM)
    # ------------------------------------------------------------------
    SchedulerCls = DDIMScheduler if args.use_ddim else DDPMScheduler
    scheduler = SchedulerCls(
        num_train_timesteps=args.num_train_timesteps,
        beta_schedule="linear",
        prediction_type="eps"                    # ε-prediction
    )

    # ------------------------------------------------------------------
    # 6. Send to device
    # ------------------------------------------------------------------
    unet = unet.to(device)
    scheduler = scheduler.to(device)
    if vae:            vae = vae.to(device)
    if class_embedder: class_embedder = class_embedder.to(device)

    # ------------------------------------------------------------------
    # 7. Load checkpoint weights
    # ------------------------------------------------------------------
    logger.info(f"Loading checkpoint from {args.ckpt}")
    load_checkpoint(unet, scheduler,
                    vae=vae,
                    class_embedder=class_embedder,
                    checkpoint_path=args.ckpt)

    # ------------------------------------------------------------------
    # 8. Build inference pipeline
    # ------------------------------------------------------------------
    pipeline = DDPMPipeline(
        unet=unet,
        scheduler=scheduler,
        vae=vae,
        class_embedder=class_embedder,
        device=device
    )
    pipeline.eval()          # just in case

    # ------------------------------------------------------------------
    # 9. Image generation
    # ------------------------------------------------------------------
    logger.info("***** Generating samples *****")
    all_fake = []            # list of tensors in [-1,1]
    batch_size = args.batch_size if hasattr(args, "batch_size") else 50

    if args.use_cfg:
        # 50 images / class, total = num_classes × 50
        for cls in tqdm(range(args.num_classes)):
            with torch.cuda.amp.autocast(dtype=torch.bfloat16 if args.bf16 else torch.float16):
                images = pipeline(
                    batch_size=batch_size,
                    class_labels=torch.full((batch_size,), cls, device=device, dtype=torch.long),
                    generator=generator,
                    guidance_scale=args.cfg_scale
                )   # returns [-1,1] torch.Tensor (B,3,H,W)
            all_fake.append(images.cpu())
            # Save grid preview
            grid = make_grid((images + 1) / 2, nrow=10)
            save_image(grid, f"outputs/sample_cls{cls:02d}.png")
    else:
        # Unconditional 5 000 images
        n_to_gen = 5000
        for _ in tqdm(range(0, n_to_gen, batch_size)):
            cur_bs = min(batch_size, n_to_gen - len(all_fake) * batch_size)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16 if args.bf16 else torch.float16):
                images = pipeline(
                    batch_size=cur_bs,
                    generator=generator,
                )
            all_fake.append(images.cpu())

    fake_images = torch.cat(all_fake, dim=0)          # [-1,1] -> (N,3,H,W)
    torch.save(fake_images, "outputs/fake_images.pt")
    logger.info(f"Generated {len(fake_images)} images.")

    # ------------------------------------------------------------------
    # 10. Load real validation images
    # ------------------------------------------------------------------
    logger.info("Loading reference images …")
    ref_images = load_reference_images(
        path=args.val_dir,
        image_size=args.unet_in_size,
        limit=len(fake_images) if not args.use_cfg else None
    )  # tensor in [0,1]
    ref_images = (ref_images * 2 - 1)                 # to [-1,1]

    # ------------------------------------------------------------------
    # 11. Evaluation (FID & IS)
    # ------------------------------------------------------------------
    logger.info("Computing FID / IS …")
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.inception import InceptionScore

    fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    is_metric  = InceptionScore(
        feature=2048, splits=10, normalize=True
    ).to(device)

    # Feed real images
    for img in tqdm(ref_images.to(device), desc="FID real"):
        fid_metric.update(img.unsqueeze(0), real=True)

    # Feed fake images
    for img in tqdm(fake_images.to(device), desc="FID fake / IS"):
        fid_metric.update(img.unsqueeze(0), real=False)
        is_metric.update(img.unsqueeze(0))

    fid_score = fid_metric.compute().item()
    is_mean, is_std = is_metric.compute()
    logger.info(f"FID: {fid_score:.3f} | IS: {is_mean:.3f} ± {is_std:.3f}")

    # ------------------------------------------------------------------
    # 12. Optional: log to Weights & Biases
    # ------------------------------------------------------------------
    if wandb.run is not None:
        wandb.log({
            "FID": fid_score,
            "IS_mean": is_mean,
            "IS_std": is_std
        })


# -------------------------------------------------
# Entry
# -------------------------------------------------
if __name__ == "__main__":
    main()
