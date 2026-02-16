"""
Recursive 1D distribution learning under real–synthetic data mixing (NN quantile-W1).

This script runs a controlled “real + synthetic memory” recursion in 1D.

At each outer iteration n:
  (1) Draw m1 new real samples from a fixed two-Gaussian mixture (the target distribution).
  (2) Train a neural sampler on the accumulated dataset (all real so far + all synthetic so far).
  (3) Generate m2 new synthetic samples from the trained sampler and add them to memory.
  (4) Evaluate convergence using exact empirical 1D Wasserstein-1 distance via sorted-quantile
      matching against a fixed reference sample from the target (when w1_compare="true").

We estimate power-law rates by regressing log(metric) on log(M_n), where M_n is the total
accumulated training size at iteration n. The main sweep reports how the empirical scaling
exponent varies with alphaReal.

"""

import os
import json
import csv
import datetime
import logging
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, gaussian_kde
from sklearn.linear_model import LinearRegression

import torch
import torch.nn as nn


# ============================================================
# 0) Matplotlib style (ECDF-style)
# ============================================================
DARK_BLUE = "#244B99"
RED = "#CC0000"


def set_vis_style():
    """
    Centralized matplotlib styling to match your ECDF-style look:
      - LaTeX-like mathtext labels/titles
      - consistent font sizes
      - subtle dashed grids (handled in plotting funcs)
    """
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            # LaTeX-like mathtext without requiring TeX
            "mathtext.fontset": "dejavusans",
            "mathtext.default": "it",
            # Deterministic-feeling defaults
            "lines.linewidth": 2.0,
            "lines.markersize": 4.0,
        }
    )


def _metric_tex(metric_name: str) -> str:
    """Convert internal metric names to mathtext consistent with ECDF-style plots."""
    up = metric_name.strip().upper()
    if up == "W1":
        return r"W_1"
    if up == "TV":
        return r"\mathrm{TV}"
    if up in ("MMD2", "MMD^2"):
        return r"\mathrm{MMD}^2"
    if up == "MMD":
        return r"\mathrm{MMD}"
    return metric_name


def _ylabel_scaling(metric_name: str) -> str:
    mt = _metric_tex(metric_name)
    return rf"$\log_{{10}}({mt})$"


def _title_scaling(metric_name: str, p_val: float, alpha_val: float) -> str:
    mt = _metric_tex(metric_name)
    return rf"${mt}$ Scaling: p={p_val}, $\alpha$={alpha_val:.2f}"


# ============================================================
# 1. Reproducibility utilities
# ============================================================
def set_all_seeds(seed: int, deterministic_torch: bool = False):
    """
    Seed python.random, NumPy, and PyTorch for reproducibility.

    deterministic_torch=True requests deterministic GPU behavior when possible.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


# ============================================================
# 2. Fixed true Gaussian mixture (no drift)
# ============================================================
w_mix = 0.35
mu1, sigma1 = -2.0, 0.8
mu2, sigma2 = 1.0, 1.3


def sample_true(n, rng):
    if n <= 0:
        return np.empty(0, dtype=float)
    z = rng.random(n)
    n1 = np.count_nonzero(z < w_mix)
    n2 = n - n1
    return np.concatenate(
        [
            rng.normal(mu1, sigma1, size=n1),
            rng.normal(mu2, sigma2, size=n2),
        ]
    )


def true_pdf(x):
    return w_mix * norm.pdf(x, mu1, sigma1) + (1 - w_mix) * norm.pdf(x, mu2, sigma2)


# Grid for density plotting only
left = min(mu1 - 6 * sigma1, mu2 - 6 * sigma2)
right = max(mu1 + 6 * sigma1, mu2 + 6 * sigma2)
m_grid = 500
xg = np.linspace(left, right, m_grid)
f_true_g = true_pdf(xg)


# ============================================================
# 3. Helpers: density-on-grid (hist or KDE), 1D empirical W1 (sorted)
# ============================================================
def density_hist_on_grid(samples, xg):
    """
    Histogram density estimate evaluated on xg centers.
    xg is treated as uniform bin centers.
    """
    samples = np.asarray(samples, dtype=float).ravel()
    if samples.size == 0:
        return np.zeros_like(xg, dtype=float)

    dx = xg[1] - xg[0]
    edges = np.concatenate(([xg[0] - 0.5 * dx], xg + 0.5 * dx))
    hist, _ = np.histogram(samples, bins=edges, density=True)
    return hist


def density_kde_on_grid(samples, xg, bw_method="scott", clip_to_grid=True):
    """
    KDE density estimate evaluated on xg.

    bw_method: 'scott', 'silverman', or a float scaling factor.
    clip_to_grid: optionally clamp samples into [xg.min, xg.max] to reduce
                  extreme-tail influence on bandwidth in small samples.
    """
    samples = np.asarray(samples, dtype=float).ravel()
    if samples.size < 2:
        return np.zeros_like(xg, dtype=float)

    if clip_to_grid:
        lo, hi = float(np.min(xg)), float(np.max(xg))
        samples = np.clip(samples, lo, hi)

    try:
        kde = gaussian_kde(samples, bw_method=bw_method)
        dens = kde.evaluate(xg)
        dens = np.maximum(dens, 0.0)
        return dens.astype(float)
    except Exception:
        return density_hist_on_grid(samples, xg)


def w1_empirical_1d_sorted(x_ref, x_model):
    """
    Exact empirical W1 in 1D for uniform weights, using sorted samples.

    If sizes differ, truncates both to n = min(n_ref, n_model).
    """
    x_ref = np.asarray(x_ref, dtype=float).ravel()
    x_model = np.asarray(x_model, dtype=float).ravel()
    n = min(x_ref.size, x_model.size)
    if n == 0:
        return np.nan
    xr = np.sort(x_ref)[:n]
    xm = np.sort(x_model)[:n]
    return float(np.mean(np.abs(xr - xm)))


# ============================================================
# 4. Neural sampler + quantile-Wasserstein training loss (1D)
# ============================================================
class QuantileNet(nn.Module):
    """
    Plain MLP sampler: u ~ Uniform(0,1) -> x in R.
    Not enforced monotone; training uses sorting-based quantile matching.
    """

    def __init__(self, hidden=(64, 64), negative_slope=0.02):
        super().__init__()
        act = lambda: nn.LeakyReLU(negative_slope)

        layers = []
        in_dim = 1
        for h in hidden:
            layers += [nn.Linear(in_dim, h), act()]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, u):
        return self.net(u)


@torch.no_grad()
def sample_from_model(model, n, device):
    if n <= 0:
        return np.empty(0, dtype=float)
    u = torch.rand(n, 1, device=device)
    x = model(u).view(-1).detach().cpu().numpy()
    return x


def make_stratified_u(b, device, jitter=1e-3):
    """
    Stratified u in (0,1) (evenly spaced) with small Gaussian jitter.
    This reduces variance relative to fresh Uniform sampling.
    """
    u = (torch.arange(b, device=device, dtype=torch.float32) + 0.5) / b
    u = u.view(-1, 1)
    if jitter and jitter > 0:
        u = (u + jitter * torch.randn_like(u)).clamp_(0.0, 1.0)
    return u


def train_model_w1_full_pass(
    model,
    X_train_np,
    device,
    epochs=1,
    batch_size=1024,
    lr=2e-4,
    weight_decay=0.0,
    grad_clip=1.0,
    log_every_epoch=False,
):
    """
    Full-coverage training:
      - Each epoch: permute full training set
      - Each minibatch: quantile-W1 loss between model samples and real minibatch
    """
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    X = torch.tensor(np.asarray(X_train_np, dtype=np.float32), device=device)
    N = X.numel()
    if N == 0:
        model.eval()
        return model

    for ep in range(epochs):
        perm = torch.randperm(N, device=device)
        ep_losses = []

        for i in range(0, N, batch_size):
            idx = perm[i : i + batch_size]
            x_real = X[idx]

            u = make_stratified_u(x_real.size(0), device=device)
            x_hat = model(u).view(-1)

            x_real_sorted, _ = torch.sort(x_real)
            x_hat_sorted, _ = torch.sort(x_hat)
            loss = torch.mean(torch.abs(x_hat_sorted - x_real_sorted))

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            ep_losses.append(loss.detach().item())

        if log_every_epoch and len(ep_losses) > 0:
            print(f"  epoch {ep+1:03d}/{epochs} | loss={float(np.mean(ep_losses)):.6f}")

    model.eval()
    return model


# ============================================================
# 5. Recursive estimator with NN (W1 metric: sorted-quantile W1, fixed reference)
# ============================================================
def run_recursive_estimator_nn(
    p_target=0.5,
    alpha=0.5,  # fraction REAL data in each new chunk
    m_total=100,  # total new data per outer iteration
    T=200,
    model_type="mlp",  # MLP only
    negative_slope=0.02,
    nn_hidden=(64, 64),
    nn_steps=10,
    nn_batch_size=1024,
    nn_lr=2e-4,
    nn_weight_decay=0.0,
    nn_grad_clip=1.0,
    # density plotting diagnostics
    grid_mc_samples=20000,
    density_method="kde",  # "kde" or "hist"
    kde_bw="scott",
    kde_clip=True,
    # W1 metric sampling
    w1_mc_samples=20000,
    w1_compare="true",  # "true" or "train"
    device=None,
    seed=None,
):
    """
    Scheme:
      - At iteration n:
          * Add m1 = round(alpha*m_total) new REAL samples
          * Add m2 = m_total - m1 new SYNTH samples (generated after training)
          * Training set = all real so far + all synthetic so far
          * Fit NN sampler by quantile-W1 training loss
      - W1 metric:
          * exact empirical 1D W1 via sorting
          * when w1_compare='true', reference sample is fixed per repetition
    """
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha must be in (0,1], got {alpha}")
    if m_total <= 0:
        raise ValueError(f"m_total must be positive, got {m_total}")
    if model_type != "mlp":
        raise ValueError("ResNet variant removed; use model_type='mlp'.")

    rng = np.random.default_rng(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Fixed per-iteration totals: split by alpha
    m1 = int(round(alpha * m_total))
    m1 = max(0, min(int(m_total), m1))
    m2 = int(m_total) - m1

    # Real fraction actually used (due to rounding)
    alpha_used = (m1 / m_total) if m_total > 0 else 1.0

    # Fixed reference for W1 if comparing to "true"
    x_true_ref = None
    if w1_compare == "true":
        x_true_ref = sample_true(w1_mc_samples, rng)

    X_real = np.empty(0, dtype=float)
    X_synth = np.empty(0, dtype=float)

    Ms = np.empty(T, dtype=int)
    W1s = np.empty(T, dtype=float)
    f_hat_history = []

    for n in range(1, T + 1):
        idx = n - 1

        # 1) Add new real data (m1 per iteration)
        x_new_real = sample_true(m1, rng)
        X_real = x_new_real if X_real.size == 0 else np.concatenate((X_real, x_new_real))

        # 2) Build training data
        X_train = X_real if X_synth.size == 0 else np.concatenate((X_real, X_synth))
        M_n = X_train.size
        Ms[idx] = M_n

        # 3) Build model (fresh init each iteration)
        model = QuantileNet(hidden=nn_hidden, negative_slope=negative_slope).to(device)

        model = train_model_w1_full_pass(
            model,
            X_train_np=X_train,
            device=device,
            epochs=nn_steps,
            batch_size=nn_batch_size,
            lr=nn_lr,
            weight_decay=nn_weight_decay,
            grad_clip=nn_grad_clip,
            log_every_epoch=False,
        )

        # 4) Generate synthetic chunk (m2 per iteration)
        if m2 > 0:
            synth_samples = sample_from_model(model, m2, device=device)
            X_synth = synth_samples if X_synth.size == 0 else np.concatenate((X_synth, synth_samples))

        # 5a) Density diagnostics on grid (for plotting only)
        x_model_mc_grid = sample_from_model(model, grid_mc_samples, device=device)
        if density_method == "kde":
            f_hat_g = density_kde_on_grid(
                x_model_mc_grid, xg, bw_method=kde_bw, clip_to_grid=kde_clip
            )
        elif density_method == "hist":
            f_hat_g = density_hist_on_grid(x_model_mc_grid, xg)
        else:
            raise ValueError(f"density_method must be 'kde' or 'hist', got {density_method}")
        f_hat_history.append(f_hat_g)

        # 5b) W1 metric vs fixed reference (or vs training set)
        if w1_compare == "train":
            x_ref = X_train
        elif w1_compare == "true":
            x_ref = x_true_ref
        else:
            raise ValueError(f"w1_compare must be 'true' or 'train', got {w1_compare}")

        # Model sample for W1
        if w1_mc_samples == grid_mc_samples:
            x_model_mc_w1 = x_model_mc_grid
        else:
            x_model_mc_w1 = sample_from_model(model, w1_mc_samples, device=device)

        W1_n = w1_empirical_1d_sorted(x_ref, x_model_mc_w1)
        W1s[idx] = W1_n

        print(
            f"[NN-W1(sorted), p={p_target:.2f}, alphaReal≈{alpha_used:.2f}] "
            f"n={n:4d} | M_n={M_n:6d} | W1={W1_n:.6f}"
        )

    return {
        "M": Ms,
        "W1": W1s,
        "p_target": float(p_target),
        "alpha_real": float(alpha_used),
        "alpha_synth": float(1.0 - alpha_used),
        "m_total": int(m_total),
        "m1": int(m1),
        "m2": int(m2),
        "f_hat_history": np.stack(f_hat_history),
        "model_type": "mlp",
        "negative_slope": float(negative_slope),
        "nn_hidden": list(nn_hidden),
        "nn_steps": int(nn_steps),
        "nn_batch_size": int(nn_batch_size),
        "nn_lr": float(nn_lr),
        "nn_weight_decay": float(nn_weight_decay),
        "nn_grad_clip": float(nn_grad_clip) if nn_grad_clip is not None else None,
        "grid_mc_samples": int(grid_mc_samples),
        "w1_mc_samples": int(w1_mc_samples),
        "w1_compare": str(w1_compare),
        "density_method": str(density_method),
        "kde_bw": str(kde_bw) if isinstance(kde_bw, str) else float(kde_bw),
        "kde_clip": bool(kde_clip),
        "device": str(device),
    }


# ============================================================
# 6. Power-law slope estimator
# ============================================================
def estimate_power_law_slope(M, vals, drop_first=10):
    M = np.asarray(M, dtype=float)
    vals = np.asarray(vals, dtype=float)

    if drop_first >= len(M) - 5:
        drop_first = max(0, len(M) - 5)

    M_use = M[drop_first:]
    vals_use = vals[drop_first:]

    logM = np.log(M_use).reshape(-1, 1)
    logV = np.log(np.maximum(vals_use, 1e-300))
    reg = LinearRegression().fit(logM, logV)
    return float(reg.coef_[0]), float(reg.intercept_)


def _make_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj


# ---------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------
if __name__ == "__main__":
    set_vis_style()

    BASE_SEED = 123456
    set_all_seeds(BASE_SEED, deterministic_torch=False)

    p_values = [0.5]
    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    T = 150
    m_total = 500
    n_reps = 50

    # Choose model (MLP only)
    MODEL_TYPE = "mlp"
    NEG_SLOPE = 0.02

    # MLP params
    NN_HIDDEN = (64, 64, 64)

    # Training knobs
    NN_STEPS = 25
    NN_BATCH = 4096
    NN_LR = 5e-4
    NN_WEIGHT_DECAY = 1e-3
    NN_GRAD_CLIP = 0

    # Density plot MC size
    GRID_MC = 20000

    # Density visualization method
    DENSITY_METHOD = "kde"
    KDE_BW = "silverman"
    KDE_CLIP = False

    # W1 metric mode
    W1_COMPARE = "true"
    W1_MC = 200000  # fixed true reference per repetition when w1_compare='true'

    # ============================================================
    # Per-repetition figure saving options
    # ============================================================
    REP_FIG_MODE = "all"      # affects W1 scaling plots
    REP_DENS_MODE = "all"     # affects density plots

    KEEP_LAST_REP_PLOTS = False

    DROP_FIRST_FOR_FIT = 20

    # -----------------------------------------------
    # 0. Output directory + logging
    # -----------------------------------------------
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"NNW1_sorted_{MODEL_TYPE}_alphaReal_T={T}_mTotal={m_total}_nreps={n_reps}_{timestamp}"
    out_dir = Path("logs/NNW1/") / run_name
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    log_file = out_dir / "run.log"
    logging.basicConfig(
        filename=log_file,
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    def log(msg: str):
        print(msg)
        logging.info(msg)

    all_results = []

    # -----------------------------------------------------------------
    # Density plot helper (ECDF-style)
    # -----------------------------------------------------------------
    def plot_density(f_true_g, f_hat_t, xg, p_val, alpha_val, rep_idx, save_path):
        plt.figure(figsize=(7, 4))
        plt.plot(xg, f_true_g, linewidth=2, label="True density")
        plt.plot(xg, f_hat_t, linewidth=2, label="Estimated density")
        plt.xlabel("x")
        plt.ylabel("density")
        plt.title(rf"Estimated density: p={p_val}, $\alpha$={alpha_val:.2f}, rep={rep_idx:03d}")
        plt.grid(True, axis="y", ls="--", alpha=0.3)
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    # -----------------------------------------------------------------
    # W1 scaling plot helper (ECDF-style)
    # -----------------------------------------------------------------
    def plot_w1_with_trend_per_rep(M, W1, p_val, alpha_val, rep_idx,
                                  drop_first=10, save_path=None):
        eps = 1e-20
        log_M = np.log10(M)
        log_val = np.log10(W1 + eps)

        plt.figure(figsize=(6, 4))
        plt.plot(log_M, log_val, "o-", alpha=0.5, markersize=3, label="Distr. Loss")

        if len(M) > drop_first + 5:
            x_fit = log_M[drop_first:]
            y_fit = log_val[drop_first:]
            slope, intercept = np.polyfit(x_fit, y_fit, 1)
            y_pred = slope * x_fit + intercept
            plt.plot(x_fit, y_pred, "r--", linewidth=2, label=f"Fit slope: {slope:.3f}")

        plt.xlabel(r"$\log_{10}$(M$_n$)")
        plt.ylabel(_ylabel_scaling("W1"))
        plt.title(_title_scaling("W1", p_val, alpha_val) + rf", rep={rep_idx:03d}")
        plt.legend(loc="upper left")
        plt.grid(True, which="both", ls="--", alpha=0.3)
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    # -----------------------------------------------------------------
    # Summary plot: alphaReal vs mean slope (ECDF-style)
    # -----------------------------------------------------------------
    def plot_alpha_vs_mean_slope(metric_key, metric_label, filename=None):
        plt.figure(figsize=(6, 4))
        metric_tex = _metric_tex(metric_label)

        unique_ps = sorted(set(res["p_target"] for res in all_results))
        for p in unique_ps:
            subset = [r for r in all_results if r["p_target"] == p]
            alpha_plot = np.array([r["alpha_real"] for r in subset])
            vals_plot = np.array([r[metric_key] for r in subset])

            order = np.argsort(alpha_plot)
            alpha_plot = alpha_plot[order]
            vals_plot = -vals_plot[order]  # keep your sign convention

            plt.plot(alpha_plot, vals_plot, "o-", label="Empirical", color=DARK_BLUE)
            theo_vals = np.minimum(p, alpha_plot)
            plt.plot(alpha_plot, theo_vals, "--", color=RED, label="Theoretical")

        plt.xlabel(r"$\alpha$")
        plt.ylabel(rf"Mean Slope ${metric_tex}$")
        plt.title(rf"Mean ${metric_tex}$ Slope vs $\alpha$")
        plt.grid(True, axis="y", ls="--", alpha=0.3)
        plt.ylim(0.0, 0.65)
        plt.yticks(np.arange(0.0, 0.7, 0.1))

        handles, labels = plt.gca().get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        plt.legend(uniq.values(), uniq.keys(), loc="upper left")
        plt.tight_layout()

        if filename is not None:
            plt.savefig(fig_dir / filename, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    # -----------------------------------------------------------------
    # Main sweeps
    # -----------------------------------------------------------------
    for alpha_real in alphas:
        log("\n" + "#" * 80)
        log(f"=== Running all p for alphaReal = {alpha_real:.2f} ===")
        log(f"    (W1 metric = exact sorted 1D empirical W1; w1_compare={W1_COMPARE})")
        log(f"    model_type={MODEL_TYPE}, neg_slope={NEG_SLOPE}")
        log(f"    m_total={m_total}")
        log(f"    wd={NN_WEIGHT_DECAY}, grad_clip={NN_GRAD_CLIP}")
        log(f"    REP_FIG_MODE={REP_FIG_MODE}, REP_DENS_MODE={REP_DENS_MODE}")
        log(f"    W1_MC={W1_MC} (fixed true reference per repetition when w1_compare='true')")

        for p in p_values:
            log("\n" + "=" * 70)
            log(f"Running recursive NN-W1 estimator {n_reps}×: p_target={p}, alphaReal={alpha_real:.2f}")

            slopes_W1 = []
            last_out = None

            for rep in range(n_reps):
                log(f"\n--- Repetition {rep + 1}/{n_reps} ---")
                rep_seed = BASE_SEED + 1000 * rep + int(1e6 * alpha_real) + int(1e3 * p)

                set_all_seeds(rep_seed, deterministic_torch=False)

                out = run_recursive_estimator_nn(
                    p_target=p,
                    alpha=alpha_real,
                    m_total=m_total,
                    T=T,
                    model_type=MODEL_TYPE,
                    negative_slope=NEG_SLOPE,
                    nn_hidden=NN_HIDDEN,
                    nn_steps=NN_STEPS,
                    nn_batch_size=NN_BATCH,
                    nn_lr=NN_LR,
                    nn_weight_decay=NN_WEIGHT_DECAY,
                    nn_grad_clip=NN_GRAD_CLIP,
                    grid_mc_samples=GRID_MC,
                    density_method=DENSITY_METHOD,
                    kde_bw=KDE_BW,
                    kde_clip=KDE_CLIP,
                    w1_mc_samples=W1_MC,
                    w1_compare=W1_COMPARE,
                    seed=rep_seed,
                )

                M_vals = out["M"]
                W1_vals = out["W1"]
                f_hist = out["f_hat_history"]

                # Per-rep density plot saving
                do_save_dens = (REP_DENS_MODE == "all") or (REP_DENS_MODE == "first" and rep == 0)
                if do_save_dens:
                    f_hat_final = f_hist[-1]
                    dens_path = fig_dir / f"density_rep={rep:03d}_p={p:.2f}_alphaReal={alpha_real:.2f}.png"
                    plot_density(f_true_g, f_hat_final, xg, p, alpha_real, rep, dens_path)

                # Optional log correction when p ≈ alphaReal (keep your convention)
                if np.isclose(p, alpha_real, atol=1e-8):
                    logM = np.log(np.maximum(M_vals, 2.0))
                    W1_for_slope = W1_vals / (logM)
                else:
                    W1_for_slope = W1_vals

                drop_first = DROP_FIRST_FOR_FIT
                slope_W1, _ = estimate_power_law_slope(M_vals, W1_for_slope, drop_first=drop_first)

                log(f"Rep {rep + 1}: slope_W1={slope_W1:.3f}")

                slopes_W1.append(slope_W1)
                last_out = out

                # Per-rep W1 scaling plot saving
                do_save_rep_fig = (REP_FIG_MODE == "all") or (REP_FIG_MODE == "first" and rep == 0)
                if do_save_rep_fig:
                    w1_rep_path = fig_dir / f"W1_rep={rep:03d}_p={p:.2f}_alphaReal={alpha_real:.2f}.png"
                    plot_w1_with_trend_per_rep(
                        M_vals, W1_vals, p, alpha_real, rep,
                        drop_first=drop_first,
                        save_path=w1_rep_path,
                    )

            slopes_W1 = np.array(slopes_W1)

            if n_reps > 1:
                mean_W1, std_W1 = float(slopes_W1.mean()), float(slopes_W1.std(ddof=1))
            else:
                mean_W1, std_W1 = float(slopes_W1.mean()), 0.0

            theo_exp = min(p, alpha_real)

            log(f"\n*** Averaged slopes over {n_reps} runs ***")
            log(f"Mean slope W1    ~ M^{mean_W1:.3f}  (std={std_W1:.3f})")
            log(f"Theoretical exponent (up to logs): min(p, alphaReal) = {theo_exp:.3f}")
            if np.isclose(p, alpha_real, atol=1e-8):
                log("Note: metrics were divided by log(M_n) before slope fitting (p ≈ alphaReal).")

            all_results.append(
                {
                    **last_out,
                    "p_target": p,
                    "alpha_real": float(alpha_real),
                    "alpha_synth": float(1.0 - alpha_real),
                    "slopes_W1": slopes_W1,
                    "mean_slope_W1": float(mean_W1),
                }
            )

    # -----------------------------------------------------------------
    # Summary plot: alphaReal vs mean slope
    # -----------------------------------------------------------------
    plot_alpha_vs_mean_slope("mean_slope_W1", "W1", filename="alphaReal_vs_mean_slope_W1.png")

    # -----------------------------------------------------------------
    # Save all_results as JSON
    # -----------------------------------------------------------------
    results_path = out_dir / "all_results.json"
    with open(results_path, "w") as f:
        json.dump(
            [{k: _make_json_serializable(v) for k, v in res.items()} for res in all_results],
            f,
            indent=2,
        )
    log(f"Saved results JSON to {results_path}")

    # -----------------------------------------------------------------
    # Save summary CSV (schema preserved)
    # -----------------------------------------------------------------
    summary_path = out_dir / "summary_slopes.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "p_target",
                "alpha_real",
                "alpha_synth",
                "m_total",
                "m1",
                "m2",
                "mean_slope_W1",
                "model_type",
            ]
        )
        for res in all_results:
            writer.writerow(
                [
                    res["p_target"],
                    res["alpha_real"],
                    res["alpha_synth"],
                    res["m_total"],
                    res["m1"],
                    res["m2"],
                    res["mean_slope_W1"],
                    res["model_type"],
                ]
            )
    log(f"Saved summary CSV to {summary_path}")
