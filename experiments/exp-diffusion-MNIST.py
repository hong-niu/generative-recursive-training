"""
Recursive real–synthetic training on MNIST using DDPM epsilon-prediction.

This script studies a fresh-batch recursive training protocol. At each outer
iteration t = 1, ..., T_steps, we construct a new training batch of fixed size
(total_per_step = m1 + m2):

  1) Sample m1 new real MNIST images (without replay; with replacement only if
     the dataset is exhausted).
  2) Sample m2 synthetic images from the current DDPM model (only for t > 1).
  3) Train the model *only* on this newly formed batch for `epochs_per_step`
     full passes (no replay buffer, no accumulated pools).

This protocol ensures that every datapoint introduced at any step receives the
same number of optimization passes (epochs_per_step), aligning with analyses
that assume equal training exposure per introduced sample.

(Optional) The primary evaluation metric is Sliced Wasserstein Distance (SWD) between
flattened real and generated images, computed via random 1D projections.

The experimental details to reproduce the paper results are stored in the 
Slurm script run-exp-diffusion-MNIST.sh

Run modes
---------
(1) Sweep mode:
    python exp-diffusion-MNIST.py

(2) Single-alpha mode (e.g., for parallel execution by alpha):
    python exp-diffusion-MNIST.py --alpha-real 0.30 --n-reps 5

Snapshotting (optional)
-----------------------
If --snapshot-interval > 0, the script saves:
  • the current training batch grid (plain)
  • the current training batch grid with synthetic tiles highlighted
  • a model-generated grid after training that step (for t > 1)

Main Dependencies
------------
pip install torch torchvision numpy matplotlib scikit-learn
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LinearRegression
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms


# ============================================================
# Defaults (argparse defaults)
# ============================================================
DEFAULT_BASE_SEED = 12345

DEFAULT_T_STEPS = 300
DEFAULT_TOTAL_PER_STEP = 1000
DEFAULT_EPOCHS_PER_STEP = 1
DEFAULT_BATCH_SIZE = 128
DEFAULT_LR = 2e-4

DEFAULT_EVAL_TRUE = 1024
DEFAULT_EVAL_GEN = 1024

# Diffusion configuration
DEFAULT_TDIFF = 200
DEFAULT_SCHEDULE = "cosine"  # {"linear", "cosine"}
DEFAULT_BETA_START = 1e-4
DEFAULT_BETA_END = 2e-2

# Model configuration
DEFAULT_BASE_CH = 64

# SWD configuration
DEFAULT_SWD_PROJECTIONS = 128
DEFAULT_SWD_MAX_SAMPLES = 1024
DEFAULT_SWD_SEED_OFFSET = 999

# Default sweep
DEFAULT_ALPHAS_REAL = [0.10, 0.30, 0.50, 0.70, 1.00]
DEFAULT_N_REPS = 5

# Snapshotting
DEFAULT_SNAPSHOT_INTERVAL = 0  # 0 disables
DEFAULT_SNAPSHOT_N = 64        # recommended perfect square (e.g., 64, 100)


# ============================================================
# Argparse + config
# ============================================================
def parse_alphas(s: str) -> List[float]:
    """Parse comma/space-separated floats."""
    parts = [p.strip() for p in s.replace(",", " ").split()]
    return [float(p) for p in parts if p]


@dataclass
class ArgsConfig:
    # randomness
    base_seed: int

    # recursion / optimization
    T_steps: int
    total_per_step: int
    epochs_per_step: int
    batch_size: int
    lr: float

    # evaluation sizes
    eval_true: int
    eval_gen: int

    # diffusion schedule
    Tdiff: int
    schedule: str
    beta_start: float
    beta_end: float

    # model size
    base_ch: int

    # SWD evaluation
    swd_projections: int
    swd_max_samples: int
    swd_seed_offset: int

    # sweep controls
    alphas_real: List[float]
    n_reps: int
    alpha_real: Optional[float]

    # snapshotting
    snapshot_interval: int
    snapshot_n: int

    # misc
    data_root: str
    device: str


def build_argparser():
    p = argparse.ArgumentParser()

    p.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)

    p.add_argument("--T-steps", type=int, default=DEFAULT_T_STEPS)
    p.add_argument("--total-per-step", type=int, default=DEFAULT_TOTAL_PER_STEP)
    p.add_argument(
        "--epochs-per-step",
        type=int,
        default=DEFAULT_EPOCHS_PER_STEP,
        help="Number of full passes over the newly formed batch (m1+m2) each outer step.",
    )
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)

    p.add_argument("--eval-true", type=int, default=DEFAULT_EVAL_TRUE)
    p.add_argument("--eval-gen", type=int, default=DEFAULT_EVAL_GEN)

    p.add_argument("--Tdiff", type=int, default=DEFAULT_TDIFF)
    p.add_argument("--schedule", type=str, default=DEFAULT_SCHEDULE, choices=["linear", "cosine"])
    p.add_argument("--beta-start", type=float, default=DEFAULT_BETA_START)
    p.add_argument("--beta-end", type=float, default=DEFAULT_BETA_END)

    p.add_argument("--base-ch", type=int, default=DEFAULT_BASE_CH)

    p.add_argument("--swd-projections", type=int, default=DEFAULT_SWD_PROJECTIONS)
    p.add_argument("--swd-max-samples", type=int, default=DEFAULT_SWD_MAX_SAMPLES)
    p.add_argument("--swd-seed-offset", type=int, default=DEFAULT_SWD_SEED_OFFSET)

    p.add_argument("--alphas-real", type=str, default=",".join(f"{a:.2f}" for a in DEFAULT_ALPHAS_REAL))
    p.add_argument("--n-reps", type=int, default=DEFAULT_N_REPS)

    # If provided, run a single alpha (reps run sequentially in the same process)
    p.add_argument("--alpha-real", type=float, default=None)

    p.add_argument("--data-root", type=str, default="")  # if empty, use <experiment_root>/data
    p.add_argument("--device", type=str, default="")     # if empty, auto

    p.add_argument(
        "--snapshot-interval",
        type=int,
        default=DEFAULT_SNAPSHOT_INTERVAL,
        help="Save snapshot grids every N outer steps (0 disables). Also saves at t=1 and t=T when enabled.",
    )
    p.add_argument(
        "--snapshot-n",
        type=int,
        default=DEFAULT_SNAPSHOT_N,
        help="Number of images in snapshot grids (perfect square recommended, e.g., 64, 100).",
    )

    return p


def cfg_from_args(a) -> ArgsConfig:
    device = a.device.strip()
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    return ArgsConfig(
        base_seed=a.base_seed,
        T_steps=a.T_steps,
        total_per_step=a.total_per_step,
        epochs_per_step=a.epochs_per_step,
        batch_size=a.batch_size,
        lr=a.lr,
        eval_true=a.eval_true,
        eval_gen=a.eval_gen,
        Tdiff=a.Tdiff,
        schedule=a.schedule,
        beta_start=a.beta_start,
        beta_end=a.beta_end,
        base_ch=a.base_ch,
        swd_projections=a.swd_projections,
        swd_max_samples=a.swd_max_samples,
        swd_seed_offset=a.swd_seed_offset,
        alphas_real=parse_alphas(a.alphas_real),
        n_reps=a.n_reps,
        alpha_real=a.alpha_real,
        snapshot_interval=a.snapshot_interval,
        snapshot_n=a.snapshot_n,
        data_root=a.data_root,
        device=device,
    )


# ============================================================
# Directories and serialization
# ============================================================
def experiment_root_dir(cfg: ArgsConfig) -> Path:
    """
    Experiment root directory encodes global settings and n_reps so downstream
    aggregation can infer the expected repetition count.
    """
    return (
        Path("logs")
        / f"mnist_4grid_fixed_seed={cfg.base_seed}"
          f"_T={cfg.T_steps}"
          f"_total={cfg.total_per_step}"
          f"_epochs={cfg.epochs_per_step}"
          f"_reps={cfg.n_reps}"
          f"_Tdiff={cfg.Tdiff}_{cfg.schedule}"
          f"_bs={cfg.batch_size}"
          f"_lr={cfg.lr:g}"
          f"_ch={cfg.base_ch}"
    )


def ensure_experiment_dirs(cfg: ArgsConfig):
    root = experiment_root_dir(cfg)
    figs_root = root / "figures"
    data_root = Path(cfg.data_root) if cfg.data_root else (root / "data")
    figs_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)
    return root, figs_root, data_root


def run_dir_for(root: Path, alpha_real: float, rep: int) -> Path:
    return root / "runs" / f"alpha_real={alpha_real:.2f}" / f"rep={rep:03d}"


def save_json(obj: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


# ============================================================
# Formatting + seeds
# ============================================================
def fmt_alpha(alpha: float, decimals: int = 2) -> str:
    return f"{float(alpha):.{decimals}f}"


def alpha_tag(alpha: float, decimals: int = 2) -> str:
    """Filesystem-safe alpha tag: 0.70 -> 0p70."""
    return fmt_alpha(alpha, decimals).replace(".", "p")


def alpha_pretty(alpha: float, decimals: int = 2) -> str:
    return f"{float(alpha):.{decimals}f}"


def run_tag(alpha: float, rep: int, decimals: int = 2) -> str:
    return f"alpha{alpha_tag(alpha, decimals)}_rep{rep:02d}"


def make_run_seed(base_seed: int, alpha_real: float, rep: int) -> int:
    """
    Construct a deterministic per-run seed from (base_seed, alpha_real, rep).

    The mapping is designed to spread seeds across alphas and repetitions while
    remaining stable across reruns.
    """
    alpha_fake = 1.0 - float(alpha_real)
    a = int(round(alpha_fake * 1000))
    return int(base_seed + a * 10000 + rep)


def seed_everything(seed: int):
    """Seed Python, NumPy, and Torch RNGs for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False


# ============================================================
# SWD metric
# ============================================================
@torch.no_grad()
def sliced_wasserstein_distance(x: torch.Tensor, y: torch.Tensor, n_proj: int = 128, seed: int = 0) -> float:
    """
    Compute Sliced Wasserstein Distance (SWD) between two point clouds.

    Parameters
    ----------
    x, y : torch.Tensor
        Tensors of shape (N, D) and (M, D). The function uses N = min(N, M) samples.
    n_proj : int
        Number of random 1D projections.
    seed : int
        Seed for projection directions.

    Returns
    -------
    float
        Mean 1D Wasserstein-1 distance over projections.
    """
    assert x.ndim == 2 and y.ndim == 2
    device = x.device
    N = min(x.size(0), y.size(0))
    x = x[:N]
    y = y[:N]

    g = torch.Generator(device=device)
    g.manual_seed(int(seed))

    proj = torch.randn(n_proj, x.size(1), device=device, generator=g)
    proj = proj / (proj.norm(dim=1, keepdim=True) + 1e-12)

    x_proj = x @ proj.t()
    y_proj = y @ proj.t()

    x_sorted, _ = torch.sort(x_proj, dim=0)
    y_sorted, _ = torch.sort(y_proj, dim=0)

    return torch.mean(torch.abs(x_sorted - y_sorted)).item()


# ============================================================
# Diffusion schedule
# ============================================================
def make_beta_schedule(
    Tdiff: int,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    device: str = "cpu",
    kind: str = "linear",
):
    """
    Construct a diffusion beta schedule and its derived alpha products.

    Returns
    -------
    betas, alphas, alpha_bars : torch.Tensor
        betas[t] in (0, 1), alphas[t] = 1 - betas[t], alpha_bars[t] = Π_{s<=t} alphas[s].
    """
    if kind == "linear":
        betas = torch.linspace(beta_start, beta_end, Tdiff, device=device)
    elif kind == "cosine":
        steps = torch.arange(Tdiff + 1, device=device, dtype=torch.float32)
        s = 0.008
        f = torch.cos(((steps / Tdiff) + s) / (1 + s) * np.pi / 2) ** 2
        alpha_bar = f / f[0]
        betas = 1 - (alpha_bar[1:] / alpha_bar[:-1]).clamp(min=1e-6, max=1.0)
        betas = betas.clamp(1e-6, 0.999)
    else:
        raise ValueError(f"Unknown schedule kind: {kind}")

    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars


# ============================================================
# UNet-style denoiser
# ============================================================
class SinusoidalTimeEmbedding(nn.Module):
    """Standard sinusoidal time embedding used in diffusion models."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        device = t.device
        freqs = torch.exp(
            -np.log(10000.0) * torch.arange(half, device=device).float() / max(half - 1, 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros((emb.size(0), 1), device=device)], dim=1)
        return emb


class ResBlock(nn.Module):
    """Residual block with time conditioning (additive bias from t-embedding)."""

    def __init__(self, ch: int, tdim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, ch)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.time = nn.Linear(tdim, ch)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time(F.silu(t_emb)).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class Down(nn.Module):
    """Stride-2 downsampling via 4x4 conv."""

    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Up(nn.Module):
    """Stride-2 upsampling via 4x4 transpose conv."""

    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SimpleUNet(nn.Module):
    """
    Lightweight UNet-style denoiser with residual blocks and time conditioning.

    Architecture:
      28x28 -> 14x14 -> 7x7 bottleneck -> 14x14 -> 28x28
    """

    def __init__(self, base_ch: int = 64, time_dim: int = 128, in_ch: int = 1):
        super().__init__()
        self.t_embed = SinusoidalTimeEmbedding(time_dim)
        self.t_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        ch1 = base_ch
        ch2 = base_ch * 2

        self.in_conv = nn.Conv2d(in_ch, ch1, 3, padding=1)

        self.rb1 = ResBlock(ch1, time_dim)
        self.down1 = Down(ch1)

        self.rb2 = ResBlock(ch1, time_dim)
        self.to_ch2 = nn.Conv2d(ch1, ch2, 1)
        self.down2 = Down(ch2)

        self.rb_mid1 = ResBlock(ch2, time_dim)
        self.rb_mid2 = ResBlock(ch2, time_dim)
        self.rb_mid3 = ResBlock(ch2, time_dim)

        self.up2 = Up(ch2)
        self.from_ch2 = nn.Conv2d(ch2, ch1, 1)
        self.rb3 = ResBlock(ch1, time_dim)

        self.up1 = Up(ch1)
        self.rb4 = ResBlock(ch1, time_dim)

        self.out_conv = nn.Conv2d(ch1, in_ch, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.t_mlp(self.t_embed(t))

        x1 = self.in_conv(x)      # (B, ch1, 28, 28)
        x1 = self.rb1(x1, t_emb)
        d1 = self.down1(x1)       # (B, ch1, 14, 14)

        d1 = self.rb2(d1, t_emb)
        skip14 = d1               # (B, ch1, 14, 14)

        d1 = self.to_ch2(d1)      # (B, ch2, 14, 14)
        d2 = self.down2(d1)       # (B, ch2, 7, 7)

        mid = self.rb_mid1(d2, t_emb)
        mid = self.rb_mid2(mid, t_emb)
        mid = self.rb_mid3(mid, t_emb)

        u2 = self.up2(mid)        # (B, ch2, 14, 14)
        u2 = self.from_ch2(u2)    # (B, ch1, 14, 14)
        u2 = u2 + skip14
        u2 = self.rb3(u2, t_emb)

        u1 = self.up1(u2)         # (B, ch1, 28, 28)
        u1 = u1 + x1
        u1 = self.rb4(u1, t_emb)

        return self.out_conv(u1)


# ============================================================
# DDPM loss + sampling
# ============================================================
def diffusion_loss_eps(model: nn.Module, x0: torch.Tensor, alpha_bars: torch.Tensor, tgen: torch.Generator) -> torch.Tensor:
    """
    Standard DDPM ε-prediction objective:
        E_{t,ε} || ε - ε_θ(x_t, t) ||^2
    """
    B = x0.size(0)
    Tdiff = alpha_bars.numel()

    t = torch.randint(0, Tdiff, (B,), device=x0.device, generator=tgen)
    eps = torch.randn(x0.shape, device=x0.device, dtype=x0.dtype, generator=tgen)

    abar = alpha_bars[t].view(B, 1, 1, 1)
    x_t = torch.sqrt(abar) * x0 + torch.sqrt(1.0 - abar) * eps

    eps_hat = model(x_t, t)
    return torch.mean((eps - eps_hat) ** 2)


@torch.no_grad()
def ddpm_sample(
    model: nn.Module,
    n: int,
    betas: torch.Tensor,
    alphas: torch.Tensor,
    alpha_bars: torch.Tensor,
    device: str,
    tgen: torch.Generator,
    shape=(1, 28, 28),
):
    """
    Standard ancestral DDPM sampling using posterior variance beta_tilde.
    """
    Tdiff = betas.numel()
    x = torch.randn(n, *shape, device=device, generator=tgen)

    for ti in reversed(range(Tdiff)):
        t = torch.full((n,), ti, device=device, dtype=torch.long)
        eps_hat = model(x, t)

        beta_t = betas[ti]
        alpha_t = alphas[ti]
        abar_t = alpha_bars[ti]
        abar_prev = alpha_bars[ti - 1] if ti > 0 else torch.tensor(1.0, device=device, dtype=alpha_bars.dtype)

        mean = (1.0 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1.0 - abar_t)) * eps_hat)

        if ti > 0:
            beta_tilde = beta_t * (1.0 - abar_prev) / (1.0 - abar_t)
            beta_tilde = beta_tilde.clamp(min=1e-20)
            z = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=tgen)
            x = mean + torch.sqrt(beta_tilde) * z
        else:
            x = mean

    return x.clamp(-1, 1)


# ============================================================
# MNIST loader
# ============================================================
def load_mnist_train(data_root: Path):
    """Load MNIST training split, mapped to [-1, 1]."""
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1),
    ])
    ds = datasets.MNIST(root=str(data_root), train=True, download=True, transform=tfm)
    return ds


# ============================================================
# Power-law helpers
# ============================================================
def estimate_powerlaw_convergence(M, Y, drop_first=10):
    """
    Fit log10(Y) ≈ a + b log10(M) on a tail segment, returning slope b.
    """
    M = np.asarray(M, float)
    Y = np.asarray(Y, float)

    drop_first = min(drop_first, max(0, len(M) - 5))
    M, Y = M[drop_first:], Y[drop_first:]

    mask = (M > 0) & np.isfinite(M) & np.isfinite(Y) & (Y > 0)
    M, Y = M[mask], Y[mask]
    if len(M) < 5:
        return np.nan, np.nan

    logM = np.log10(M).reshape(-1, 1)
    logY = np.log10(Y)

    reg = LinearRegression().fit(logM, logY)
    return float(reg.coef_[0]), float(reg.intercept_)


# ============================================================
# Plot helpers
# ============================================================
def plot_curves(M, metric, loss, out_prefix, fig_dir: Path):
    """Save SWD-vs-M and training-loss curves."""
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.plot(M, metric, "-o", markersize=3)
    plt.xlabel("M (cumulative total samples added)")
    plt.ylabel("SWD (lower is better)")
    plt.title("Evaluation SWD vs M")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / f"{out_prefix}_swd_vs_M.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(1, len(loss) + 1), loss, "-o", markersize=3)
    plt.xlabel("Outer step")
    plt.ylabel("Training loss (ε-MSE)")
    plt.title("Training loss over recursion")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / f"{out_prefix}_train_loss.png", dpi=200)
    plt.close()


@torch.no_grad()
def save_grid_samples(model, betas, alphas, alpha_bars, device, seed, out_path: Path, n=64):
    """Generate and save an n-image grid from the current model."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    tgen = torch.Generator(device=device)
    tgen.manual_seed(int(seed))

    x = ddpm_sample(
        model, n=n, betas=betas, alphas=alphas, alpha_bars=alpha_bars,
        device=device, tgen=tgen, shape=(1, 28, 28)
    ).cpu()

    x = (x + 1) / 2
    x = x.clamp(0, 1)

    k = int(np.sqrt(n))
    fig, axes = plt.subplots(k, k, figsize=(6, 6))
    for i in range(k):
        for j in range(k):
            axes[i, j].imshow(x[i * k + j, 0], cmap="gray")
            axes[i, j].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


@torch.no_grad()
def save_grid_from_tensor_batch(
    x: torch.Tensor,
    out_path: Path,
    n=64,
    is_synth: Optional[torch.Tensor] = None,
):
    """
    Save an n-image grid from a tensor batch.

    Parameters
    ----------
    x : torch.Tensor
        Shape (N, 1, 28, 28), values in [-1, 1] or [0, 1].
    is_synth : Optional[torch.Tensor]
        Boolean mask of length N; if provided, synthetic tiles are outlined.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if x.ndim != 4:
        raise ValueError(f"Expected 4D tensor (N,C,H,W); got {x.shape}")

    N = int(x.size(0))
    if N <= 0:
        return

    n = min(int(n), N)
    k = int(np.sqrt(n))
    k = max(k, 1)
    n = k * k

    x = x[:n].detach().cpu()
    if is_synth is not None:
        is_synth = is_synth[:n].detach().cpu().to(torch.bool)

    if x.min() < 0:
        x = (x + 1) / 2
    x = x.clamp(0, 1)

    fig, axes = plt.subplots(k, k, figsize=(6, 6))
    for i in range(k):
        for j in range(k):
            idx = i * k + j
            ax = axes[i, j]
            ax.imshow(x[idx, 0], cmap="gray")
            ax.axis("off")

            if is_synth is not None and bool(is_synth[idx].item()):
                rect = patches.Rectangle(
                    (0, 0), 1, 1,
                    transform=ax.transAxes,
                    fill=False,
                    edgecolor="#AF69EE",
                    linewidth=5,
                )
                ax.add_patch(rect)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


# ============================================================
# One run (given alpha_real, rep)
# ============================================================
def run_one(alpha_real: float, rep: int, cfg: ArgsConfig, mnist_ds, root: Path) -> Dict[str, Any]:
    """
    Run a single (alpha_real, rep) experiment, saving outputs under:
        <root>/runs/alpha_real=XX.XX/rep=YYY/
    """
    m1 = int(round(alpha_real * cfg.total_per_step))
    m2 = cfg.total_per_step - m1

    run_seed = make_run_seed(cfg.base_seed, alpha_real, rep)
    tag = run_tag(alpha_real, rep, decimals=2)

    rdir = run_dir_for(root, alpha_real, rep)
    fdir = rdir / "figures"
    snapshots_dir = fdir / "snapshots"
    rdir.mkdir(parents=True, exist_ok=True)
    fdir.mkdir(parents=True, exist_ok=True)

    run_cfg = asdict(cfg)
    run_cfg.update({"alpha_real": float(alpha_real), "rep": int(rep), "run_seed": int(run_seed)})
    save_json(run_cfg, rdir / "config.json")

    out = run_recursive_diffusion_mnist_freshbatch(
        mnist_ds=mnist_ds,
        T=cfg.T_steps,
        m1=m1,
        m2=m2,
        epochs_per_step=cfg.epochs_per_step,
        lr=cfg.lr,
        batch_size=cfg.batch_size,
        eval_true=cfg.eval_true,
        eval_gen=cfg.eval_gen,
        device=cfg.device,
        seed=run_seed,
        Tdiff=cfg.Tdiff,
        schedule_kind=cfg.schedule,
        beta_start=cfg.beta_start,
        beta_end=cfg.beta_end,
        base_ch=cfg.base_ch,
        swd_projections=cfg.swd_projections,
        swd_max_samples=cfg.swd_max_samples,
        swd_seed_offset=cfg.swd_seed_offset,
        snapshot_interval=cfg.snapshot_interval,
        snapshot_n=cfg.snapshot_n,
        snapshots_dir=snapshots_dir,
        run_seed=run_seed,
        alpha_real=float(alpha_real),
        rep=int(rep),
    )

    np.savez(
        rdir / "metrics.npz",
        M=out["M"],
        metric=out["metric"],
        loss_history=out["loss_history"],
        alpha_real=float(alpha_real),
        rep=int(rep),
        seed=int(run_seed),
    )

    plot_curves(out["M"], out["metric"], out["loss_history"], out_prefix=tag, fig_dir=fdir)

    save_grid_samples(
        out["denoiser"], out["betas"], out["alphas"], out["alpha_bars"],
        device=cfg.device, seed=run_seed + 555, out_path=fdir / f"{tag}_samples.png", n=64
    )

    # Slope estimate retained for compatibility (may be ill-posed if M is nearly linear/regular).
    M = out["M"]
    metric_raw = out["metric"].copy()
    metric_for_slope = metric_raw / np.log(M) if np.isclose(alpha_real, 0.5) else metric_raw
    slope, intercept = estimate_powerlaw_convergence(M, metric_for_slope, drop_first=10)

    (rdir / "slope.txt").write_text(f"{slope:.10f}\n")
    (rdir / "slope_intercept.txt").write_text(f"{intercept:.10f}\n")

    print(f"[{tag}] alpha_real={alpha_real:.2f} rep={rep} seed={run_seed} slope={slope:.6f}", flush=True)
    return {"alpha_real": alpha_real, "rep": rep, "seed": run_seed, "slope": slope, "run_dir": str(rdir)}


# ============================================================
# Recursive training loop (core): fresh-batch-only
# ============================================================
def run_recursive_diffusion_mnist_freshbatch(
    mnist_ds,
    T: int = 200,
    m1: int = 200,
    m2: int = 200,
    epochs_per_step: int = 1,
    lr: float = 2e-4,
    batch_size: int = 128,
    eval_true: int = 1024,
    eval_gen: int = 1024,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 1234,
    Tdiff: int = 200,
    schedule_kind: str = "cosine",
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    base_ch: int = 64,
    swd_projections: int = 128,
    swd_max_samples: int = 1024,
    swd_seed_offset: int = 999,
    snapshot_interval: int = 0,
    snapshot_n: int = 64,
    snapshots_dir: Optional[Path] = None,
    run_seed: int = 0,
    alpha_real: float = 0.0,
    rep: int = 0,
) -> Dict[str, Any]:
    """
    Core recursion loop implementing fresh-batch-only training.

    Returns a dict containing:
      - M: cumulative samples added
      - metric: SWD curve over outer steps
      - loss_history: training loss per outer step
      - denoiser and diffusion schedule tensors
    """
    seed_everything(seed)

    rng = np.random.default_rng(seed)
    rng_eval = np.random.default_rng(seed + 101)

    tgen = torch.Generator(device=device)
    tgen.manual_seed(seed + 202)

    model = SimpleUNet(base_ch=base_ch, time_dim=128, in_ch=1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    betas, alphas, alpha_bars = make_beta_schedule(
        Tdiff=Tdiff, beta_start=beta_start, beta_end=beta_end, device=device, kind=schedule_kind
    )

    # Real index stream for selecting new real samples (replacement used only after exhaustion)
    N = len(mnist_ds)
    all_idx = np.arange(N, dtype=int)
    rng.shuffle(all_idx)
    ptr = 0
    exhausted_warned = False
    used_real = set()

    M_arr = np.zeros(T, dtype=int)
    eval_metric = np.zeros(T, dtype=float)
    loss_history = np.zeros(T, dtype=float)

    # Fixed evaluation subset (real)
    eval_idx = rng_eval.choice(N, size=eval_true, replace=False).tolist()
    real_eval = torch.stack([mnist_ds[i][0] for i in eval_idx], dim=0)
    real_eval_flat = real_eval.view(real_eval.size(0), -1).to(device)

    swd_seed = int(seed + swd_seed_offset)

    def maybe_snapshot(step_batch_cpu: torch.Tensor, is_synth: torch.Tensor, tstep: int) -> bool:
        """
        Optionally save snapshot grids:
          - training batch (plain)
          - training batch (synthetic highlighted)
          - model-generated grid after training (tstep > 1)
        """
        do_snap = (
            snapshots_dir is not None
            and int(snapshot_interval) > 0
            and (tstep == 1 or tstep == T or (tstep % int(snapshot_interval) == 0))
        )
        if not do_snap:
            return False

        snap_dir = Path(snapshots_dir)
        snap_dir.mkdir(parents=True, exist_ok=True)
        tag = f"alpha{alpha_pretty(alpha_real)}_rep{rep:03d}_t={tstep:04d}_seed={run_seed}"

        perm = rng.permutation(step_batch_cpu.size(0))
        perm_t = torch.from_numpy(perm).long()

        x_show = step_batch_cpu[perm_t]
        is_synth_show = is_synth[perm_t]

        save_grid_from_tensor_batch(
            x_show,
            snap_dir / f"{tag}_train_plain.png",
            n=snapshot_n,
            is_synth=None,
        )
        save_grid_from_tensor_batch(
            x_show,
            snap_dir / f"{tag}_train_highlight.png",
            n=snapshot_n,
            is_synth=is_synth_show,
        )

        del x_show, is_synth_show

        if tstep > 1:
            model.eval()
            with torch.no_grad():
                snap_gen = torch.Generator(device=device)
                snap_gen.manual_seed(int(seed + 700000 + tstep))
                xg_snap = ddpm_sample(
                    model=model,
                    n=int(snapshot_n),
                    betas=betas,
                    alphas=alphas,
                    alpha_bars=alpha_bars,
                    device=device,
                    tgen=snap_gen,
                    shape=(1, 28, 28),
                ).cpu()
            save_grid_from_tensor_batch(xg_snap, snap_dir / f"{tag}_gen.png", n=snapshot_n)
            del xg_snap

        return True

    epochs = max(1, int(epochs_per_step))

    for tstep in range(1, T + 1):
        # ----------------------------------------------------
        # 1) Construct NEW batch: m1 new real + m2 new synthetic (t > 1)
        # ----------------------------------------------------
        picked: List[int] = []
        if m1 > 0:
            while len(picked) < m1 and ptr < N:
                idx = int(all_idx[ptr])
                ptr += 1
                if idx not in used_real:
                    used_real.add(idx)
                    picked.append(idx)

            if len(picked) < m1:
                if not exhausted_warned:
                    print(
                        f"[warn] Exhausted unique MNIST indices at t={tstep}. "
                        f"Filling remaining real samples with replacement.",
                        flush=True,
                    )
                    exhausted_warned = True
                need = m1 - len(picked)
                picked.extend(rng.choice(N, size=need, replace=True).tolist())

        real_new = None
        if len(picked) > 0:
            real_new = torch.stack([mnist_ds[i][0] for i in picked], dim=0)  # CPU, in [-1, 1]

        synth_new = None
        if tstep > 1 and m2 > 0:
            model.eval()
            with torch.no_grad():
                synth_new = ddpm_sample(
                    model=model,
                    n=m2,
                    betas=betas,
                    alphas=alphas,
                    alpha_bars=alpha_bars,
                    device=device,
                    tgen=tgen,
                    shape=(1, 28, 28),
                ).cpu()

        parts: List[torch.Tensor] = []
        is_synth_list: List[bool] = []

        if real_new is not None:
            parts.append(real_new)
            is_synth_list.extend([False] * int(real_new.size(0)))

        if synth_new is not None:
            parts.append(synth_new)
            is_synth_list.extend([True] * int(synth_new.size(0)))

        if len(parts) == 0:
            M_arr[tstep - 1] = 0
            loss_history[tstep - 1] = np.nan
            eval_metric[tstep - 1] = np.nan
            continue

        step_batch = torch.cat(parts, dim=0)  # CPU
        is_synth = torch.tensor(is_synth_list, dtype=torch.bool)

        added = int(step_batch.size(0))  # equals m1 + m2 (except t=1 -> m1 when synth absent)
        prev = int(M_arr[tstep - 2]) if tstep > 1 else 0
        M_arr[tstep - 1] = prev + added

        _ = maybe_snapshot(step_batch, is_synth, tstep)

        # ----------------------------------------------------
        # 2) Train ONLY on step_batch for `epochs` full passes
        # ----------------------------------------------------
        model.train()
        running = 0.0
        n_steps = 0

        ds_step = TensorDataset(step_batch)
        loader = DataLoader(
            ds_step,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=(device == "cuda"),
        )

        for _ep in range(epochs):
            for (x0_cpu,) in loader:
                x0 = x0_cpu.to(device=device, dtype=torch.float32, non_blocking=True)

                loss = diffusion_loss_eps(model, x0, alpha_bars, tgen)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                running += float(loss.item())
                n_steps += 1
                del x0, loss

        loss_history[tstep - 1] = running / max(n_steps, 1)

        # ----------------------------------------------------
        # 3) Evaluate via SWD against a fixed real_eval subset
        # ----------------------------------------------------
        model.eval()
        with torch.no_grad():
            g_eval = ddpm_sample(
                model=model,
                n=eval_gen,
                betas=betas,
                alphas=alphas,
                alpha_bars=alpha_bars,
                device=device,
                tgen=tgen,
                shape=(1, 28, 28),
            )

        g_eval_flat = g_eval.view(g_eval.size(0), -1)
        n_eval = min(swd_max_samples, real_eval_flat.size(0), g_eval_flat.size(0))

        metric = sliced_wasserstein_distance(
            real_eval_flat[:n_eval],
            g_eval_flat[:n_eval],
            n_proj=swd_projections,
            seed=swd_seed + tstep,
        )
        eval_metric[tstep - 1] = metric

        print(
            f"[t={tstep:4d}] M_cum={M_arr[tstep-1]:7d} | M_new={added:5d} | "
            f"SWD={eval_metric[tstep-1]:.6f} | train_loss={loss_history[tstep-1]:.6f} | "
            f"epochs={epochs}",
            flush=True,
        )

        del g_eval, g_eval_flat, step_batch, real_new, synth_new, parts, is_synth
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    return {
        "M": M_arr,
        "metric": eval_metric,
        "loss_history": loss_history,
        "denoiser": model,
        "betas": betas,
        "alphas": alphas,
        "alpha_bars": alpha_bars,
        "seed": seed,
    }


# ============================================================
# Main runner
# ============================================================
def main():
    cfg = cfg_from_args(build_argparser().parse_args())

    root, figs_root, data_root = ensure_experiment_dirs(cfg)

    save_json(asdict(cfg), root / "config_experiment.json")

    print(f"Device: {cfg.device}")
    print(f"Experiment root: {root}")
    print(f"Figures root:    {figs_root}")
    print(f"Data root:       {data_root}")

    if cfg.alpha_real is None:
        print(f"Mode: SWEEP (alphas={cfg.alphas_real}, n_reps={cfg.n_reps})", flush=True)
    else:
        print(f"Mode: SINGLE-ALPHA (alpha_real={cfg.alpha_real:.2f}, reps=0..{cfg.n_reps-1})", flush=True)

    print(f"Fresh-batch training: epochs_per_step={cfg.epochs_per_step}", flush=True)

    if cfg.snapshot_interval > 0:
        print(f"Snapshots: every {cfg.snapshot_interval} steps (and t=1,t=T), snapshot_n={cfg.snapshot_n}", flush=True)
        print("Snapshot training grids saved as: *_train_plain.png and *_train_highlight.png", flush=True)
    else:
        print("Snapshots: disabled (snapshot_interval=0)", flush=True)

    mnist_ds = load_mnist_train(data_root)

    alphas = [float(cfg.alpha_real)] if cfg.alpha_real is not None else list(cfg.alphas_real)

    for alpha_real in alphas:
        m1 = int(round(alpha_real * cfg.total_per_step))
        m2 = cfg.total_per_step - m1

        print("\n" + "#" * 70)
        print(f"alpha_real={alpha_real:.2f}  (m1={m1}, m2={m2})")
        print("#" * 70, flush=True)

        for rep in range(cfg.n_reps):
            _ = run_one(alpha_real, rep, cfg, mnist_ds, root)

    print("\nDone. Per-run outputs are under:")
    print(root / "runs", flush=True)


if __name__ == "__main__":
    main()
