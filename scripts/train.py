#!/usr/bin/env python3
"""Training script for Wan2.2 video world model + action expert (flow matching)."""

import argparse
import contextlib
import json as _json
import logging
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from einops import rearrange
from tqdm import tqdm
import torchvision.transforms as T
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from models.wan_2_2_models.transformers.model import WanModel
from models.wan_2_2_models.text_encoder.t5 import T5EncoderModel
from models.wan_2_2_models.vae.vae2_2 import Wan2_2_VAE
from torch.optim.lr_scheduler import LambdaLR
from utils.model_utils import load_checkpoints, count_model_parameters

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with a low-rank adapter: y = Wx + (xA)(B) * scale."""

    def __init__(self, linear: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.linear = linear
        self.lora_A = nn.Parameter(torch.randn(linear.in_features, rank) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(rank, linear.out_features))
        self.scale  = alpha / rank
        self.drop   = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        linear.requires_grad_(False)

    def forward(self, x):
        return self.linear(x) + self.drop(x) @ self.lora_A @ self.lora_B * self.scale


def inject_lora_video_backbone(model, rank: int, alpha: float, dropout: float = 0.0):
    """Replace attention linear layers in model.blocks with LoRA wrappers."""
    def wrap(linear):
        return LoRALinear(linear, rank, alpha, dropout)

    for block in model.blocks:
        sa = block.self_attn
        ca = block.cross_attn

        if getattr(sa, 'fused_qkv', False) and sa.qkv is not None:
            sa.qkv = wrap(sa.qkv)
        else:
            if sa.q is not None: sa.q = wrap(sa.q)
            if sa.k is not None: sa.k = wrap(sa.k)
            if sa.v is not None: sa.v = wrap(sa.v)
        sa.o = wrap(sa.o)

        if getattr(ca, 'fused_qkv', False) and ca.qkv is not None:
            ca.qkv = wrap(ca.qkv)
        else:
            if ca.q is not None: ca.q = wrap(ca.q)
            if ca.k is not None: ca.k = wrap(ca.k)
            if ca.v is not None: ca.v = wrap(ca.v)
        ca.o = wrap(ca.o)

    logger.info(f"Injected LoRA (rank={rank}, alpha={alpha}) into {len(model.blocks)} video blocks")


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def make_dataset(
    dataset_path: str,
    img_size,
    num_frames: int,
    action_chunk: int,
    episodes: list = None,
) -> LeRobotDataset:
    root = Path(dataset_path)
    repo_id = f"{root.parent.name}/{root.name}"
    _meta = LeRobotDataset(repo_id, root=root)
    fps = _meta.fps
    camera_keys = list(_meta.meta.camera_keys)

    delta_timestamps = {
        cam_key: [i / fps for i in range(num_frames)]
        for cam_key in camera_keys
    }
    delta_timestamps["state"]   = [0.0]
    delta_timestamps["actions"] = [i / fps for i in range(action_chunk)]

    image_transforms = T.Compose([
        T.Resize(img_size),
        T.Lambda(lambda x: x * 2.0 - 1.0),
    ])

    return LeRobotDataset(
        repo_id, root=root, episodes=episodes,
        delta_timestamps=delta_timestamps,
        image_transforms=image_transforms,
    )


def make_slim_dataset(
    dataset_path: str,
    action_chunk: int,
    episodes: list = None,
) -> LeRobotDataset:
    """Actions/states only — no image loading."""
    root = Path(dataset_path)
    repo_id = f"{root.parent.name}/{root.name}"
    _meta = LeRobotDataset(repo_id, root=root)
    fps = _meta.fps
    delta_timestamps = {
        "state":   [0.0],
        "actions": [i / fps for i in range(action_chunk)],
    }
    return LeRobotDataset(repo_id, root=root, episodes=episodes, delta_timestamps=delta_timestamps)


def collate_fn(batch, camera_keys, action_dim):
    images = torch.stack([
        torch.stack(
            [sample[cam].permute(1, 0, 2, 3) for cam in camera_keys],
            dim=1,
        )
        for sample in batch
    ])
    return {
        "images":  images,
        "texts":   [sample["task"] for sample in batch],
        "actions": torch.stack([sample["actions"][:, :action_dim] for sample in batch]),
        "states":  torch.stack([sample["state"][:, :action_dim]   for sample in batch]),
    }


# ---------------------------------------------------------------------------
# Pre-caching — batched DataLoader, not a per-sample Python loop
# ---------------------------------------------------------------------------

def precompute_encodings(
    dataset, camera_keys, vae, text_encoder, cache_dir, device, param_dtype,
    encode_batch_size: int = 16,
    num_workers: int = 8,
):
    """Encode all samples with VAE + T5 once; resumes safely if interrupted."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    missing = [i for i in range(len(dataset)) if not (cache_dir / f"{i}.pt").exists()]
    if not missing:
        logger.info(f"Latent cache complete ({len(dataset)} samples) at {cache_dir}")
        return

    logger.info(
        f"Pre-computing latents for {len(missing)}/{len(dataset)} samples "
        f"(encode_batch={encode_batch_size}, workers={num_workers}) → {cache_dir}"
    )

    V      = len(camera_keys)
    subset = torch.utils.data.Subset(dataset, missing)
    loader = DataLoader(
        subset,
        batch_size=encode_batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        collate_fn=lambda b: b,
    )

    written = 0
    for batch_samples in tqdm(loader, desc="Encoding latents"):
        B = len(batch_samples)
        with torch.no_grad():
            all_views = torch.stack([
                batch_samples[b][cam].permute(1, 0, 2, 3)
                for b in range(B)
                for cam in camera_keys
            ]).to(device, dtype=param_dtype, non_blocking=True)

            enc_flat = vae.encode(list(all_views.unbind(0)))

            texts   = [batch_samples[b]["task"] for b in range(B)]
            ctx_all = text_encoder(texts, device)

        for b in range(B):
            enc_b = enc_flat[b * V : (b + 1) * V]
            z_stk = torch.stack(enc_b, dim=1)
            z     = rearrange(z_stk, "c v t h w -> c t h (v w)").cpu()
            torch.save(
                {"z": z, "context": ctx_all[b].cpu(), "num_views": V},
                cache_dir / f"{missing[written]}.pt",
            )
            written += 1

    logger.info(f"Pre-computation complete — {written} samples written to {cache_dir}")


class CachedLatentDataset(torch.utils.data.Dataset):
    """Serves pre-computed VAE latents + T5 embeddings alongside actions/states."""

    def __init__(self, base_dataset, cache_dir: str, action_dim: int):
        self.base       = base_dataset
        self.cache_dir  = Path(cache_dir)
        self.action_dim = action_dim
        manifest = self.cache_dir / "valid_indices.json"
        if manifest.exists():
            self.valid_indices = _json.loads(manifest.read_text())
        else:
            print("CachedLatentDataset: scanning cache for shape consistency (one-time)...")
            first = torch.load(self.cache_dir / "0.pt", map_location="cpu", weights_only=True)
            expected_tz = first["z"].shape[1]
            self.valid_indices = [
                i for i in range(len(base_dataset))
                if torch.load(self.cache_dir / f"{i}.pt", map_location="cpu", weights_only=True)["z"].shape[1] == expected_tz
            ]
            manifest.write_text(_json.dumps(self.valid_indices))
            dropped = len(base_dataset) - len(self.valid_indices)
            if dropped:
                print(f"CachedLatentDataset: dropped {dropped} samples with unexpected T_z (boundary frames)")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        sample = self.base[real_idx]
        cached = torch.load(self.cache_dir / f"{real_idx}.pt", map_location="cpu", weights_only=True)
        return {
            "z":         cached["z"],
            "context":   cached["context"],
            "num_views": cached["num_views"],
            "actions":   sample["actions"][:, :self.action_dim],
            "states":    sample["state"][:, :self.action_dim],
            "task":      sample["task"],
        }


def collate_fn_cached(batch):
    # Stack z and context into tensors here in the worker so the main thread
    # gets a single contiguous tensor — one .to(device) call instead of B small ones
    return {
        "z":         torch.stack([s["z"]       for s in batch]),   # [B, C, T_z, H_z, W_z*V]
        "context":   torch.stack([s["context"] for s in batch]),   # [B, seq_len, text_dim]
        "num_views": batch[0]["num_views"],
        "actions":   torch.stack([s["actions"] for s in batch]),
        "states":    torch.stack([s["states"]  for s in batch]),
        "texts":     [s["task"]    for s in batch],
    }


# ---------------------------------------------------------------------------
# Flow matching helpers
# ---------------------------------------------------------------------------

def sample_flow_sigma(batch_size: int, device: torch.device) -> torch.Tensor:
    """Logit-normal sigma ∈ (0,1)."""
    return torch.sigmoid(torch.randn(batch_size, device=device))


def flow_noisy(x_clean: torch.Tensor, noise: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    s = sigma.view(-1, *([1] * (x_clean.dim() - 1)))
    return (1.0 - s) * x_clean + s * noise


# tok_mask is constant for a given model config — cache it across steps
_TOK_MASK_CACHE: dict = {}

def build_video_batch(z_batch, sigma_vid, t_vid, patch_size, seq_len, device):
    """
    Fully vectorized: build noisy latents + per-token timestep for the whole batch.
    z_batch: [B, C, T_z, H_z, W_z*V] — already stacked, already on device.
    Returns: noisy_z_list, noise_batch, mask2_batch, tok_mask, video_timestep
    """
    B           = z_batch.shape[0]
    noise_batch = torch.randn_like(z_batch)

    # mask2: 0 for first temporal slice (conditioning frame), 1 for the rest
    mask2_batch          = torch.ones_like(z_batch)
    mask2_batch[:, :, 0] = 0.0

    s           = sigma_vid.view(B, 1, 1, 1, 1)
    noisy_batch = (1.0 - s) * z_batch + s * noise_batch
    noisy_batch = (1.0 - mask2_batch) * z_batch + mask2_batch * noisy_batch

    # tok_mask is identical for every sample and every step — compute once and cache
    cache_key = (patch_size[1], patch_size[2], seq_len, z_batch.shape[2])
    if cache_key not in _TOK_MASK_CACHE:
        tok = mask2_batch[0, 0, :, ::patch_size[1], ::patch_size[2]].flatten()
        if seq_len > tok.size(0):
            tok = torch.cat([tok, tok.new_ones(seq_len - tok.size(0))])
        _TOK_MASK_CACHE[cache_key] = tok
    tok_mask = _TOK_MASK_CACHE[cache_key]

    video_timestep = t_vid.unsqueeze(1).float() * tok_mask.unsqueeze(0)   # [B, seq_len]

    return list(noisy_batch.unbind(0)), noise_batch, mask2_batch, tok_mask, video_timestep


# ---------------------------------------------------------------------------
# Checkpoint saving
# ---------------------------------------------------------------------------

def save_checkpoint(accelerator, model, args, global_step: int):
    if not accelerator.is_main_process:
        return
    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
    os.makedirs(save_path, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    action_prefixes = ("action_proj_in", "action_blocks", "action_time_embedding",
                       "action_time_projection", "action_head")
    def _keep(k):
        return any(k.startswith(p) for p in action_prefixes) or (
            args.train_video and (k.startswith("blocks") or k.startswith("head"))
        )
    state = {k: v for k, v in unwrapped.state_dict().items() if _keep(k)}
    torch.save(state, os.path.join(save_path, "action_weights.pt"))
    lora_state = {k: v for k, v in unwrapped.state_dict().items()
                  if "lora_A" in k or "lora_B" in k}
    if lora_state:
        torch.save(lora_state, os.path.join(save_path, "lora_weights.pt"))
    logger.info(f"Saved checkpoint at step {global_step} → {save_path}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def denoise_and_decode(
    model, vae, z_cond, mask2_b, context, seq_len,
    num_views, device, param_dtype, patch_size, num_train_timesteps, num_steps=20,
):
    x  = (1.0 - mask2_b) * z_cond + mask2_b * torch.randn_like(z_cond)
    dt = 1.0 / num_steps
    for sigma in torch.linspace(1.0, dt, num_steps, device=device):
        t_int    = (sigma * num_train_timesteps).long().clamp(1, num_train_timesteps - 1).item()
        tok_mask = mask2_b[0, :, ::patch_size[1], ::patch_size[2]].flatten()
        video_ts = (tok_mask * t_int).unsqueeze(0)
        if seq_len > video_ts.size(1):
            video_ts = torch.cat(
                [video_ts, video_ts.new_ones(1, seq_len - video_ts.size(1)) * t_int], dim=1
            )
        out = model([x], t=video_ts, context=context, seq_len=seq_len,
                    return_video=True, return_action=False, store_buffer=False)
        x   = x - dt * out["video"][0]
        x   = (1.0 - mask2_b) * z_cond + mask2_b * x

    x_views = rearrange(x.float(), "c t h (v w) -> v c t h w", v=num_views)
    decoded = vae.decode(list(x_views.unbind(0)))
    frames  = [((f.clamp(-1,1)+1)/2*255).byte().permute(1,2,3,0).cpu().numpy() for f in decoded]
    return np.concatenate(frames, axis=2)


@torch.no_grad()
def run_validation(
    accelerator, model, vae, text_encoder,
    val_dataloader, camera_keys, action_dim, action_chunk,
    device, param_dtype, patch_size, num_train_timesteps,
    global_step, log_with, num_denoise_steps=20, using_cache=False,
    action_mean=None, action_std=None,
):
    if not accelerator.is_main_process:
        return

    unwrapped = accelerator.unwrap_model(model)
    unwrapped.eval()

    total_loss  = 0.0
    num_batches = 0
    vis_data    = None

    for batch in val_dataloader:
        # Single .to() call for the whole stacked tensor
        actions = batch["actions"].to(device, dtype=param_dtype, non_blocking=True)
        states  = batch["states"].to(device,  dtype=param_dtype, non_blocking=True)
        if action_mean is not None:
            actions = (actions - action_mean) / action_std
            states  = (states  - action_mean) / action_std
        B = actions.shape[0]

        if using_cache:
            z_batch  = batch["z"].to(device, dtype=param_dtype, non_blocking=True)
            ctx_batch = batch["context"].to(device, dtype=param_dtype, non_blocking=True)
            z_list   = list(z_batch.unbind(0))
            context  = list(ctx_batch.unbind(0))
            num_views = batch["num_views"]
        else:
            images    = batch["images"].to(device, dtype=param_dtype, non_blocking=True)
            num_views = images.shape[2]
            z_list    = []
            for b in range(B):
                views = list(images[b].unbind(dim=1))
                enc   = vae.encode(views)
                z_stk = torch.stack(enc, dim=1)
                z_list.append(rearrange(z_stk, "c v t h w -> c t h (v w)"))
            context = text_encoder(batch["texts"], device)

        z_batch_gpu = torch.stack(z_list)
        T_z, H_z, W_z = z_batch_gpu.shape[2], z_batch_gpu.shape[3], z_batch_gpu.shape[4]
        seq_len = T_z * H_z * W_z // (patch_size[1] * patch_size[2])

        sigma_vid = sample_flow_sigma(B, device)
        t_vid     = (sigma_vid * num_train_timesteps).long().clamp(1, num_train_timesteps - 1)
        noisy_z_list, noise_batch, mask2_batch, tok_mask, video_timestep = build_video_batch(
            z_batch_gpu, sigma_vid, t_vid, patch_size, seq_len, device
        )

        sigma_act       = sample_flow_sigma(B, device)
        t_act           = (sigma_act * num_train_timesteps).long().clamp(1, num_train_timesteps - 1)
        noise_action    = torch.randn_like(actions)
        noisy_actions   = flow_noisy(actions, noise_action, sigma_act)
        action_timestep = t_act.unsqueeze(1).expand(-1, action_chunk).float()

        out = unwrapped(
            noisy_z_list, t=video_timestep, context=context, seq_len=seq_len,
            action_states=noisy_actions, action_timestep=action_timestep,
            history_action_state=states,
            return_video=False, return_action=True, store_buffer=True,
        )
        loss = F.mse_loss(out["action"].float(), (noise_action - actions).float())
        total_loss  += loss.item()
        num_batches += 1

        if vis_data is None:
            vis_data = (
                z_list[0].clone(), mask2_batch[0].clone(),
                [context[0]], seq_len, num_views,
            )

    val_loss = total_loss / max(1, num_batches)
    logger.info(f"step={global_step:6d}  val_loss={val_loss:.4f}")
    log_dict = {"val/loss": val_loss}

    if log_with and vis_data is not None:
        import wandb
        z_cond, mask2_b, ctx, s_len, n_views = vis_data
        logger.info("Generating validation video …")
        frames = denoise_and_decode(
            unwrapped, vae, z_cond, mask2_b, ctx, s_len,
            n_views, device, param_dtype, patch_size, num_train_timesteps, num_denoise_steps,
        )
        log_dict["val/video"] = wandb.Video(frames.transpose(0,3,1,2), fps=4, format="mp4")

    if log_with:
        accelerator.log(log_dict, step=global_step)

    unwrapped.train()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train Wan2.2 action expert")
    parser.add_argument("--config",       type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir",   type=str, default=None)
    parser.add_argument("--latent_cache_dir", type=str, default="/lambda/nfs/hfm/david/tau0wm")
    parser.add_argument("--encode_batch_size", type=int, default=16,
                        help="VAE+T5 batch size for the one-time precompute step.")
    parser.add_argument("--encode_workers",    type=int, default=8,
                        help="DataLoader workers for the precompute step.")
    parser.add_argument("--train_video",  action="store_true")
    parser.add_argument("--lora_rank",    type=int,   default=8)
    parser.add_argument("--lora_alpha",   type=float, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--action_loss_weight", type=float, default=1.0)
    parser.add_argument("--video_loss_weight",  type=float, default=1.0)
    parser.add_argument("--batch_size",                  type=int,   default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int,   default=4)
    parser.add_argument("--num_train_epochs",            type=int,   default=50)
    parser.add_argument("--max_train_steps",             type=int,   default=None)
    parser.add_argument("--lr",                          type=float, default=5e-5)
    parser.add_argument("--weight_decay",                type=float, default=1e-2)
    parser.add_argument("--lr_warmup_steps",             type=int,   default=1000)
    parser.add_argument("--max_grad_norm",               type=float, default=1.0)
    parser.add_argument("--num_train_timesteps",         type=int,   default=1000)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--mixed_precision", type=str, default="bf16",
                        choices=["no", "fp16", "bf16"])
    parser.add_argument("--num_workers",    type=int, default=4)
    parser.add_argument("--log_every",      type=int, default=10)
    parser.add_argument("--save_every",     type=int, default=500)
    parser.add_argument("--steps_per_epoch", type=int, default=None)
    parser.add_argument("--val_episodes",   type=int, default=10)
    parser.add_argument("--val_every",      type=int, default=None)
    parser.add_argument("--val_denoise_steps", type=int, default=20)
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--wandb_project",  type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    from datetime import datetime
    args = parse_args()
    if args.output_dir is None:
        now = datetime.now()
        tag = (args.wandb_project or "run").replace("-", "")
        args.output_dir = f"outputs/train/{now.strftime('%Y-%m-%d')}/{now.strftime('%H-%M-%S')}_{tag}"

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    action_dim   = cfg["action_dim"]
    action_chunk = cfg["action_chunk"]
    img_size     = tuple(cfg["img_size"])
    num_frames   = cfg.get("chunk", 9)
    vae_path     = cfg["vae_path"]
    t5_cfg       = cfg["text_encoder"]
    diff_cfg     = cfg["diffusion_model"]
    model_config = diff_cfg["config"]
    model_path   = diff_cfg["model_path"]
    patch_size   = model_config.get("patch_size", [1, 2, 2])
    val_every    = args.val_every or args.save_every

    # ------------------------------------------------------------------
    # Accelerator — must come first; accelerate's logger requires it
    # ------------------------------------------------------------------
    log_with = "wandb" if args.wandb_project else None
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=log_with,
        project_config=ProjectConfiguration(
            project_dir=args.output_dir,
            logging_dir=os.path.join(args.output_dir, "logs"),
        ),
    )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    set_seed(args.seed + accelerator.process_index)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        if log_with:
            accelerator.init_trackers(
                args.wandb_project,
                config={**vars(args), **cfg},
                init_kwargs={"wandb": {"name": args.wandb_run_name}},
            )

    device      = accelerator.device
    param_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32

    # Action normalisation stats (after Accelerator so logger is ready)
    stats_path = cfg.get("statistics_file")
    if stats_path and Path(stats_path).exists():
        with open(stats_path) as f:
            _stats = _json.load(f)
        action_mean = torch.tensor(_stats["action"]["mean"][:action_dim], dtype=torch.float32)
        action_std  = torch.tensor(_stats["action"]["std"][:action_dim],  dtype=torch.float32)
        logger.info(f"Loaded action stats from {stats_path}")
    else:
        logger.warning("No statistics_file in config — actions will NOT be normalised!")
        action_mean = torch.zeros(action_dim, dtype=torch.float32)
        action_std  = torch.ones(action_dim,  dtype=torch.float32)

    action_mean = action_mean.to(device, dtype=param_dtype)
    action_std  = action_std.to(device,  dtype=param_dtype)

    # ------------------------------------------------------------------
    # Frozen models
    # ------------------------------------------------------------------
    logger.info("Loading VAE …")
    vae = Wan2_2_VAE(vae_pth=vae_path, dtype=param_dtype, device=device)

    logger.info("Loading T5 …")
    text_encoder = T5EncoderModel(
        text_len=t5_cfg.get("text_len", 512),
        dtype=param_dtype, device=device,
        checkpoint_path=t5_cfg["checkpoint_path"],
        tokenizer_path=t5_cfg["tokenizer_path"],
    )

    # ------------------------------------------------------------------
    # WanModel
    # ------------------------------------------------------------------
    logger.info("Loading WanModel …")
    model = WanModel(**model_config)
    load_checkpoints(model, model_path, strict=False)

    model.requires_grad_(False)
    for mod in [model.action_proj_in, model.action_blocks,
                model.action_time_embedding, model.action_time_projection,
                model.action_head]:
        mod.requires_grad_(True)

    if args.lora_rank > 0:
        assert not args.train_video, "--lora_rank and --train_video are mutually exclusive"
        lora_alpha = args.lora_alpha if args.lora_alpha is not None else float(args.lora_rank)
        inject_lora_video_backbone(model, args.lora_rank, lora_alpha, args.lora_dropout)
        for block in model.blocks:
            for m in block.modules():
                if isinstance(m, LoRALinear):
                    m.lora_A.requires_grad_(True)
                    m.lora_B.requires_grad_(True)
    elif args.train_video:
        for mod in [model.blocks, model.head]:
            mod.requires_grad_(True)

    if args.gradient_checkpointing:
        model.gradient_checkpointing = True

    total_params, trainable_params = count_model_parameters(model)
    logger.info(f"Total: {total_params/1e9:.2f}B | Trainable: {trainable_params/1e6:.1f}M")

    # LoRA needs grads through the backbone; fully frozen does not
    backbone_needs_grad = args.train_video or (args.lora_rank > 0)
    # Build the context once — not inside the hot loop
    fwd_ctx = contextlib.nullcontext() if backbone_needs_grad else torch.no_grad()

    # ------------------------------------------------------------------
    # Dataset — train / val split by episode
    # ------------------------------------------------------------------
    _tmp = LeRobotDataset(
        f"{Path(args.dataset_path).parent.name}/{Path(args.dataset_path).name}",
        root=Path(args.dataset_path),
    )
    total_episodes = _tmp.meta.total_episodes
    val_ep_ids   = list(range(total_episodes - args.val_episodes, total_episodes))
    train_ep_ids = list(range(0, total_episodes - args.val_episodes))
    logger.info(f"Train episodes: {len(train_ep_ids)} | Val episodes: {len(val_ep_ids)}")

    # ------------------------------------------------------------------
    # Pre-compute VAE + T5 (batched DataLoader, runs once then cached)
    # ------------------------------------------------------------------
    using_cache = args.latent_cache_dir is not None

    if using_cache:
        full_train  = make_dataset(args.dataset_path, img_size, num_frames, action_chunk,
                                   episodes=train_ep_ids)
        full_val    = make_dataset(args.dataset_path, img_size, num_frames, action_chunk,
                                   episodes=val_ep_ids)
        camera_keys = list(full_train.meta.camera_keys)

        if accelerator.is_main_process:
            logger.info("=== Pre-computing VAE + T5 encodings (runs once) ===")
            precompute_encodings(
                full_train, camera_keys, vae, text_encoder,
                os.path.join(args.latent_cache_dir, "train"),
                device, param_dtype,
                encode_batch_size=args.encode_batch_size,
                num_workers=args.encode_workers,
            )
            precompute_encodings(
                full_val, camera_keys, vae, text_encoder,
                os.path.join(args.latent_cache_dir, "val"),
                device, param_dtype,
                encode_batch_size=args.encode_batch_size,
                num_workers=args.encode_workers,
            )
        accelerator.wait_for_everyone()

        slim_train    = make_slim_dataset(args.dataset_path, action_chunk, episodes=train_ep_ids)
        slim_val      = make_slim_dataset(args.dataset_path, action_chunk, episodes=val_ep_ids)
        train_dataset = CachedLatentDataset(
            slim_train, os.path.join(args.latent_cache_dir, "train"), action_dim
        )
        val_dataset = CachedLatentDataset(
            slim_val, os.path.join(args.latent_cache_dir, "val"), action_dim
        )
        train_collate = collate_fn_cached
        val_collate   = collate_fn_cached
        logger.info("Using cached latents — VAE and T5 will not run during training.")
    else:
        train_dataset = make_dataset(args.dataset_path, img_size, num_frames, action_chunk,
                                     episodes=train_ep_ids)
        val_dataset   = make_dataset(args.dataset_path, img_size, num_frames, action_chunk,
                                     episodes=val_ep_ids)
        camera_keys   = list(train_dataset.meta.camera_keys)
        train_collate = lambda b: collate_fn(b, camera_keys, action_dim)
        val_collate   = lambda b: collate_fn(b, camera_keys, action_dim)

    _dl_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    train_dataloader = DataLoader(
        train_dataset, shuffle=True,  drop_last=True,
        collate_fn=train_collate, **_dl_kwargs
    )
    val_dataloader = DataLoader(
        val_dataset,   shuffle=False, drop_last=False,
        collate_fn=val_collate,   **_dl_kwargs
    )
    logger.info(
        f"Train: {len(train_dataset)} samples | Val: {len(val_dataset)} samples | "
        f"cameras: {camera_keys}"
    )

    # ------------------------------------------------------------------
    # Optimiser + cosine LR schedule with warmup
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
        fused=True,   # fused AdamW kernel — free ~5% speedup on CUDA
    )
    steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_steps       = args.max_train_steps or args.num_train_epochs * steps_per_epoch

    def _lr_lambda(step):
        if step < args.lr_warmup_steps:
            return float(step) / max(1, args.lr_warmup_steps)
        progress = float(step - args.lr_warmup_steps) / max(1, max_steps - args.lr_warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    lr_scheduler = LambdaLR(optimizer, _lr_lambda)

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    num_train_timesteps = args.num_train_timesteps

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    global_step = 0
    logger.info(f"Training for up to {max_steps} gradient steps …")
    logger.info(
        "Mode: " + (
            "train_video (full backbone)" if args.train_video else
            f"LoRA rank={args.lora_rank}" if args.lora_rank > 0 else
            "action-expert only (backbone frozen)"
        )
    )

    for epoch in range(args.num_train_epochs):
        model.train()
        epoch_loss = 0.0

        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch}",
                          disable=not accelerator.is_main_process):
            with accelerator.accumulate(model):

                # Single non-blocking transfer for each tensor
                actions = batch["actions"].to(device, dtype=param_dtype, non_blocking=True)
                states  = batch["states"].to(device,  dtype=param_dtype, non_blocking=True)
                actions = (actions - action_mean) / action_std
                states  = (states  - action_mean) / action_std
                B       = actions.shape[0]

                if using_cache:
                    # collate_fn_cached already stacked → one .to() instead of B small ones
                    z_batch_gpu  = batch["z"].to(device, dtype=param_dtype, non_blocking=True)
                    ctx_batch_gpu = batch["context"].to(device, dtype=param_dtype, non_blocking=True)
                    z_list   = list(z_batch_gpu.unbind(0))
                    context  = list(ctx_batch_gpu.unbind(0))
                else:
                    images = batch["images"].to(device, dtype=param_dtype, non_blocking=True)
                    with torch.no_grad():
                        z_list = []
                        for b in range(B):
                            views = list(images[b].unbind(dim=1))
                            enc   = vae.encode(views)
                            z_stk = torch.stack(enc, dim=1)
                            z_list.append(rearrange(z_stk, "c v t h w -> c t h (v w)"))
                        context = text_encoder(batch["texts"], device)
                    z_batch_gpu = torch.stack(z_list)

                T_z, H_z, W_z = z_batch_gpu.shape[2], z_batch_gpu.shape[3], z_batch_gpu.shape[4]
                seq_len = T_z * H_z * W_z // (patch_size[1] * patch_size[2])

                # Fully vectorized noise + timestep
                sigma_vid = sample_flow_sigma(B, device)
                t_vid     = (sigma_vid * num_train_timesteps).long().clamp(1, num_train_timesteps - 1)
                # Pass the already-stacked z_batch_gpu — no re-stacking inside
                noisy_z_list, noise_batch, mask2_batch, tok_mask, video_timestep = build_video_batch(
                    z_batch_gpu, sigma_vid, t_vid, patch_size, seq_len, device
                )

                sigma_act       = sample_flow_sigma(B, device)
                t_act           = (sigma_act * num_train_timesteps).long().clamp(1, num_train_timesteps - 1)
                noise_action    = torch.randn_like(actions)
                noisy_actions   = flow_noisy(actions, noise_action, sigma_act)
                action_timestep = t_act.unsqueeze(1).expand(-1, action_chunk).float()

                # Single fused forward pass
                # fwd_ctx built once before the loop — no Python object creation per step
                with fwd_ctx:
                    out = model(
                        noisy_z_list, t=video_timestep, context=context, seq_len=seq_len,
                        action_states=noisy_actions, action_timestep=action_timestep,
                        history_action_state=states,
                        return_video=args.train_video,
                        return_action=True,
                        store_buffer=not args.train_video,
                    )

                action_loss = F.mse_loss(out["action"].float(), (noise_action - actions).float())

                if args.train_video:
                    v_pred_batch = torch.stack(out["video"]).float()
                    v_target     = (noise_batch - z_batch_gpu).float()
                    ph, pw       = patch_size[1], patch_size[2]
                    v_pred_tok   = rearrange(v_pred_batch,                  "b c t h w -> b (t h w) c")
                    v_target_tok = rearrange(v_target[:, :, :, ::ph, ::pw], "b c t h w -> b (t h w) c")
                    n   = min(v_pred_tok.size(1), tok_mask.size(0))
                    fut = tok_mask[:n].bool()
                    video_loss = (
                        F.mse_loss(
                            v_pred_tok[:, :n][:, fut].reshape(-1, v_pred_tok.size(-1)),
                            v_target_tok[:, :n][:, fut].reshape(-1, v_target_tok.size(-1)),
                        ) if fut.any() else torch.tensor(0.0, device=device)
                    )
                    loss = args.action_loss_weight * action_loss + args.video_loss_weight * video_loss
                else:
                    video_loss = None
                    loss = args.action_loss_weight * action_loss

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], args.max_grad_norm
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)   # free grad memory immediately

            if accelerator.sync_gradients:
                global_step += 1
                epoch_loss  += loss.detach().item()

                if global_step % args.log_every == 0 and accelerator.is_main_process:
                    loss_val = loss.detach().item()
                    lr_val   = lr_scheduler.get_last_lr()[0]
                    logger.info(f"step={global_step:6d}  loss={loss_val:.4f}  lr={lr_val:.2e}")
                    if log_with:
                        log_dict = {
                            "train/loss":        loss_val,
                            "train/action_loss": action_loss.item(),
                            "train/lr":          lr_val,
                        }
                        if video_loss is not None:
                            log_dict["train/video_loss"] = video_loss.item()
                        accelerator.log(log_dict, step=global_step)

                if global_step % args.save_every == 0:
                    save_checkpoint(accelerator, model, args, global_step)

                if global_step % val_every == 0:
                    run_validation(
                        accelerator, model, vae, text_encoder,
                        val_dataloader, camera_keys, action_dim, action_chunk,
                        device, param_dtype, patch_size, num_train_timesteps,
                        global_step, log_with,
                        num_denoise_steps=args.val_denoise_steps,
                        using_cache=using_cache,
                        action_mean=action_mean, action_std=action_std,
                    )
                    model.train()

                if global_step >= max_steps:
                    break
                if args.steps_per_epoch and (global_step % args.steps_per_epoch == 0) and global_step > 0:
                    break

        logger.info(f"Epoch {epoch} — avg loss: {epoch_loss / max(1, steps_per_epoch):.4f}")
        if global_step >= max_steps:
            break

    save_checkpoint(accelerator, model, args, global_step)
    if log_with:
        accelerator.end_training()


if __name__ == "__main__":
    main()
