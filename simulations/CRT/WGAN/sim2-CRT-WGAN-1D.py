"""
Recursive distribution learning under real–synthetic data mixing using a neural sampler.

We study a recursive estimator in which a neural network sampler is repeatedly
trained on an accumulated mixture of real and synthetic samples. At each
iteration n:

1. A fixed batch of m1 new real samples is drawn from a stationary
   two-component Gaussian mixture.
2. The training set is formed by combining all accumulated real samples
   with all previously generated synthetic samples.
3. A neural sampler maps u ~ Uniform(0,1) to samples in R and is trained
   using a sorting-based quantile Wasserstein-1 loss.
4. A new batch of synthetic samples is generated from the trained neural
   sampler and added to the synthetic pool.

The mixing proportion is controlled by alpha_real ∈ (0, 1], interpreted as
the fraction of new samples per iteration that come from the true real
distribution. In the implementation, alpha_real determines

    - m1 = round(alpha_real * m_total)
    - m2 = m_total - m1,


where m1 is the number of new real samples and m2 is the number of new
synthetic samples added at each iteration.

We evaluate convergence to the ground-truth distribution using:

    - empirical Wasserstein-1 distance (W1), computed from sorted samples
    - maximum mean discrepancy (MMD), computed from sample-based unbiased
      MMD^2 with an RBF kernel

Because the unbiased finite-sample estimator of MMD^2 can be negative due to
sampling variability, raw MMD^2 values are stored separately. For the MMD
error curve, only strictly positive MMD^2 estimates are square-rooted; nonpositive
or nonfinite estimates are recorded as NaN and excluded from log–log slope
fitting.

For visualization only, a kernel density estimate of the final neural
sampler distribution is computed at the end of each repetition. This KDE is
not used to compute W1 or MMD.

Power-law scaling exponents are estimated by log–log regression of the
distributional error against the accumulated sample size M_n. Empirical
slopes are compared to the theoretical exponent min(p, alpha_real), up to
logarithmic corrections in the critical regime p = alpha_real.
"""


#!/usr/bin/env python3
import os
import json
import csv
import datetime
import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, gaussian_kde
from sklearn.linear_model import LinearRegression

import torch
import torch.nn as nn
import random


def set_all_seeds(seed: int, deterministic_torch: bool = False):
    """
    Seed numpy + torch (+ python random) for reproducibility.
    deterministic_torch=True makes GPU behavior more repeatable but can be slower.
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
# 1. Fixed true Gaussian mixture (no drift)
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


# Grid ONLY for plotting / storing a final KDE curve (not for metrics)
left = min(mu1 - 6 * sigma1, mu2 - 6 * sigma2)
right = max(mu1 + 6 * sigma1, mu2 + 6 * sigma2)
m_grid = 1000
xg = np.linspace(left, right, m_grid)
f_true_g = true_pdf(xg)  # analytic truth curve


# ============================================================
# 2. Helpers: KDE-on-grid for plotting, 1D empirical W1 (sorted)
# ============================================================
def _renorm_density_on_grid(f, xg):
    f = np.asarray(f, dtype=float)
    area = float(np.trapz(f, xg))
    if not np.isfinite(area) or area <= 1e-12:
        return np.zeros_like(f, dtype=float)
    return f / area


def density_kde_on_grid(samples, xg, bw_method="scott", clip_to_grid=True):
    """
    KDE density estimate evaluated on xg (for plotting only).
    """
    samples = np.asarray(samples, dtype=float).ravel()
    if samples.size < 2:
        return np.zeros_like(xg, dtype=float)

    if clip_to_grid:
        lo, hi = float(np.min(xg)), float(np.max(xg))
        samples = np.clip(samples, lo, hi)

    kde = gaussian_kde(samples, bw_method=bw_method)
    dens = kde.evaluate(xg)
    dens = np.maximum(dens, 0.0)
    return _renorm_density_on_grid(dens.astype(float), xg)


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
# 3. Empirical MMD^2 from samples (correct kernel MMD; no KDE)
# ============================================================
def _rbf_kernel_sqdist(a, b, sigma):
    # a: (n,1), b: (m,1)
    sq = (a - b.T) ** 2
    return np.exp(-sq / (2.0 * sigma * sigma))

def mmd2_biased_rbf_1d(x, y, sigma):
    """
    Biased empirical MMD^2 with RBF kernel.
    Includes diagonal terms, so it is nonnegative up to numerical precision.
    Better for convergence curves and log-slope plots.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1, 1)
    y = np.asarray(y, dtype=np.float64).reshape(-1, 1)

    n = x.shape[0]
    m = y.shape[0]
    if n < 1 or m < 1:
        return np.nan

    Kxx = _rbf_kernel_sqdist(x, x, sigma)
    Kyy = _rbf_kernel_sqdist(y, y, sigma)
    Kxy = _rbf_kernel_sqdist(x, y, sigma)

    return float(Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean())

def mmd2_unbiased_rbf_1d(x, y, sigma):
    """
    Unbiased empirical MMD^2 with RBF kernel.
    O(n^2 + m^2 + nm). Use subsampling to control cost.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1, 1)
    y = np.asarray(y, dtype=np.float64).reshape(-1, 1)

    n = x.shape[0]
    m = y.shape[0]
    if n < 2 or m < 2:
        return np.nan

    Kxx = _rbf_kernel_sqdist(x, x, sigma)
    Kyy = _rbf_kernel_sqdist(y, y, sigma)
    Kxy = _rbf_kernel_sqdist(x, y, sigma)

    sum_xx = Kxx.sum() - np.trace(Kxx)
    sum_yy = Kyy.sum() - np.trace(Kyy)

    term_xx = sum_xx / (n * (n - 1))
    term_yy = sum_yy / (m * (m - 1))
    term_xy = 2.0 * Kxy.mean()

    return float(term_xx + term_yy - term_xy)


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
# 5. Recursive estimator
#    - W1: exact sorted empirical W1
#    - MMD: sqrt(unbiased empirical MMD^2) from samples (NO KDE)
#    - KDE: computed ONCE per repetition (end), stored as f_hat_history shape (1,m_grid)
# ============================================================
def run_recursive_estimator_nn(
    p_target=0.5,
    alpha=0.5,  # fraction REAL data in each new chunk
    m_total=100,  # total new data per outer iteration
    T=200,
    model_type="mlp",
    negative_slope=0.02,
    nn_hidden=(64, 64),
    nn_steps=10,
    nn_batch_size=1024,
    nn_lr=2e-4,
    nn_weight_decay=0.0,
    nn_grad_clip=1.0,
    # metric sampling
    w1_mc_samples=20000,
    w1_compare="true",  # "true" or "train"
    # MMD settings (sample-based)
    mmd_sigma=0.5,
    mmd_mc_subsample=2000,  # subsample for MMD to keep O(n^2) manageable
    # KDE for plotting only (computed once at end)
    do_kde_for_plot=True,
    kde_plot_samples=20000,
    kde_bw="scott",
    kde_clip=True,
    device=None,
    seed=None,
):
    if model_type != "mlp":
        raise ValueError(f"ResNet removed: model_type must be 'mlp', got {model_type}")
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha must be in (0,1], got {alpha}")
    if m_total <= 0:
        raise ValueError(f"m_total must be positive, got {m_total}")

    rng = np.random.default_rng(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    m1 = int(round(alpha * m_total))
    m1 = max(0, min(int(m_total), m1))
    m2 = int(m_total) - m1
    alpha_used = (m1 / m_total) if m_total > 0 else 1.0

    # Fixed reference for W1/MMD if comparing to true
    x_true_ref = None
    if w1_compare == "true":
        x_true_ref = sample_true(w1_mc_samples, rng)

    X_real = np.empty(0, dtype=float)
    X_synth = np.empty(0, dtype=float)

    Ms = np.empty(T, dtype=int)
    W1s = np.empty(T, dtype=float)
    MMDs = np.empty(T, dtype=float)
    MMD2s = np.empty(T, dtype=float)

    last_model = None

    for n in range(1, T + 1):
        idx = n - 1

        # 1) Add new real data
        x_new_real = sample_true(m1, rng)
        X_real = x_new_real if X_real.size == 0 else np.concatenate((X_real, x_new_real))

        # 2) Training data
        X_train = X_real if X_synth.size == 0 else np.concatenate((X_real, X_synth))
        M_n = X_train.size
        Ms[idx] = M_n

        # 3) Build model (fresh init each iteration)
        model = QuantileNet(hidden=nn_hidden, negative_slope=negative_slope).to(device)

        # try it out 
        # dynamic_batch_size = min(M_n, 50000) # or even M_n if your GPU memory permits

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
        last_model = model

        # 4) Generate synthetic chunk
        if m2 > 0:
            synth_samples = sample_from_model(model, m2, device=device)
            X_synth = synth_samples if X_synth.size == 0 else np.concatenate((X_synth, synth_samples))

        # 5) W1 metric
        if w1_compare == "train":
            x_ref = X_train
        elif w1_compare == "true":
            x_ref = x_true_ref
        else:
            raise ValueError(f"w1_compare must be 'true' or 'train', got {w1_compare}")

        x_model_mc = sample_from_model(model, w1_mc_samples, device=device)

        W1_n = w1_empirical_1d_sorted(x_ref, x_model_mc)
        W1s[idx] = W1_n

        # 6) MMD metric (sample-based)
        # subsample both sides to size s (no replacement) for O(s^2)
        s = int(mmd_mc_subsample) if mmd_mc_subsample is not None else 0
        if s >= 2 and np.isfinite(W1_n):
            s_eff = min(s, len(x_ref), len(x_model_mc))
            if s_eff >= 2:
                if len(x_ref) == s_eff:
                    xr = np.asarray(x_ref, dtype=float)
                else:
                    xr = rng.choice(np.asarray(x_ref, dtype=float), size=s_eff, replace=False)

                if len(x_model_mc) == s_eff:
                    xm = np.asarray(x_model_mc, dtype=float)
                else:
                    xm = rng.choice(np.asarray(x_model_mc, dtype=float), size=s_eff, replace=False)

                mmd2_val = mmd2_unbiased_rbf_1d(xr, xm, sigma=mmd_sigma)
            else:
                mmd2_val = np.nan
        else:
            mmd2_val = np.nan

        MMD2s[idx] = mmd2_val
        # For the unbiased MMD^2 estimator, finite-sample estimates can be <= 0.
        # Those are invalid for sqrt/log-slope fitting, so keep them as NaN.
        if np.isfinite(mmd2_val) and mmd2_val > 0:
            MMDs[idx] = float(np.sqrt(mmd2_val))
        else:
            MMDs[idx] = np.nan

        print(
            f"[NN-W1(sorted), p={p_target:.2f}, alphaReal≈{alpha_used:.2f}] "
            f"n={n:4d} | M_n={M_n:6d} | W1={W1_n:.6f} | MMD={MMDs[idx]:.6f}"
        )

    # KDE for plotting ONLY (computed once per repetition, at the end)
    f_hat_history = None
    if do_kde_for_plot and last_model is not None:
        x_model_plot = sample_from_model(last_model, kde_plot_samples, device=device)
        f_hat_final = density_kde_on_grid(x_model_plot, xg, bw_method=kde_bw, clip_to_grid=kde_clip)
        # Keep compatibility with "f_hat_history[-1]" consumers:
        f_hat_history = np.stack([f_hat_final], axis=0)

    return {
        "M": Ms,
        "W1": W1s,
        "MMD": MMDs,
        "MMD2": MMD2s,
        "p_target": float(p_target),
        "alpha_real": float(alpha_used),
        "alpha_synth": float(1.0 - alpha_used),
        "m_total": int(m_total),
        "m1": int(m1),
        "m2": int(m2),
        "f_hat_history": f_hat_history,  # shape (1, m_grid) or None
        "model_type": "mlp",
        "negative_slope": float(negative_slope),
        "nn_hidden": list(nn_hidden),
        "res_width": None,
        "res_blocks": None,
        "nn_steps": int(nn_steps),
        "nn_batch_size": int(nn_batch_size),
        "nn_lr": float(nn_lr),
        "nn_weight_decay": float(nn_weight_decay),
        "nn_grad_clip": float(nn_grad_clip) if nn_grad_clip is not None else None,
        "w1_mc_samples": int(w1_mc_samples),
        "w1_compare": str(w1_compare),
        "mmd_sigma": float(mmd_sigma),
        "mmd_mc_subsample": int(mmd_mc_subsample),
        "do_kde_for_plot": bool(do_kde_for_plot),
        "kde_plot_samples": int(kde_plot_samples),
        "kde_bw": str(kde_bw) if isinstance(kde_bw, str) else float(kde_bw),
        "kde_clip": bool(kde_clip),
        "device": str(device),
        # store grid so remake scripts can re-plot without ambiguity
        "grid": {"left": float(left), "right": float(right), "m_grid": int(m_grid)},
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

    # Drop invalid values before log-slope fitting.
    # This is especially important for MMD from unbiased MMD^2, where
    # negative/zero finite-sample estimates are stored as NaN after sqrt.
    mask = np.isfinite(M_use) & np.isfinite(vals_use) & (M_use > 0) & (vals_use > 0)
    M_use = M_use[mask]
    vals_use = vals_use[mask]

    if M_use.size < 3:
        raise ValueError("Not enough valid positive points to fit power-law slope.")

    logM = np.log(M_use).reshape(-1, 1)
    logV = np.log(vals_use)
    reg = LinearRegression().fit(logM, logV)
    return float(reg.coef_[0]), float(reg.intercept_)


def _make_json_serializable(obj):
    import numpy as _np

    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    if isinstance(obj, (_np.float32, _np.float64)):
        return float(obj)
    if isinstance(obj, (_np.int32, _np.int64)):
        return int(obj)
    return obj


# ---------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------
if __name__ == "__main__":
    BASE_SEED = 123456
    set_all_seeds(BASE_SEED, deterministic_torch=False)

    p_values = [0.5]

    alphas = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    Ts = [150] * 10
    DROP_FIRST_FOR_FIT = [20] * 10
    assert len(alphas) == len(Ts) == len(DROP_FIRST_FOR_FIT), (
        "alphas, Ts, and DROP_FIRST_FOR_FIT must have the same length/order"
    )

    m_total = 500 
    n_reps = 100

    MODEL_TYPE = "mlp"
    NEG_SLOPE = 0.02
    NN_HIDDEN = (64, 64, 64)



    NN_STEPS = 25
    NN_BATCH = 4096 
    NN_LR = 5e-4 
    NN_WEIGHT_DECAY = 1e-3 
    NN_GRAD_CLIP = 0

    # W1 metric mode
    W1_COMPARE = "true"
    W1_MC = 200000 # was 200000

    # MMD from samples: subsample size for O(s^2)
    MMD_SIGMA = 0.5
    MMD_MC_SUBSAMPLE = 3000 

    # KDE for plotting only
    KDE_PLOT_SAMPLES = 20000
    KDE_BW = "silverman"
    KDE_CLIP = False

    # Per-repetition figure saving options
    REP_FIG_MODE = "all"   # affects scaling plots (W1 + MMD)
    REP_DENS_MODE = "all"  # "all" or "first"
    KEEP_LAST_REP_PLOTS = False  # kept

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"NNW1_sorted_{MODEL_TYPE}_alphaReal_Tvar_mTotal={m_total}_nreps={n_reps}_MMDsample_KDEplot_{timestamp}"
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

    def log_config(title: str, cfg: dict):
        log("\n" + "-" * 80)
        log(title)
        log(json.dumps(cfg, indent=2, sort_keys=True))
        log("-" * 80 + "\n")

    RUN_CONFIG = {
        "base_seed": BASE_SEED,
        "p_values": p_values,
        "alphas": alphas,
        "Ts": Ts,
        "drop_first_for_fit": DROP_FIRST_FOR_FIT,
        "m_total": m_total,
        "n_reps": n_reps,
        "model_type": MODEL_TYPE,
        "negative_slope": NEG_SLOPE,
        "nn_hidden": list(NN_HIDDEN),
        "nn_steps": NN_STEPS,
        "nn_batch_size": NN_BATCH,
        "nn_lr": NN_LR,
        "nn_weight_decay": NN_WEIGHT_DECAY,
        "nn_grad_clip": NN_GRAD_CLIP,
        "w1_compare": W1_COMPARE,
        "w1_mc_samples": W1_MC,
        "mmd_sigma": MMD_SIGMA,
        "mmd_mc_subsample": MMD_MC_SUBSAMPLE,
        "kde_for_plot_only": {
            "kde_plot_samples": KDE_PLOT_SAMPLES,
            "kde_bw": KDE_BW,
            "kde_clip": KDE_CLIP,
        },
        "grid": {"left": float(left), "right": float(right), "m_grid": int(m_grid)},
        "true_mixture": {
            "w_mix": float(w_mix),
            "mu1": float(mu1),
            "sigma1": float(sigma1),
            "mu2": float(mu2),
            "sigma2": float(sigma2),
        },
        "rep_fig_mode": REP_FIG_MODE,
        "rep_dens_mode": REP_DENS_MODE,
        "keep_last_rep_plots": KEEP_LAST_REP_PLOTS,
    }
    log_config("RUN CONFIG", RUN_CONFIG)

    all_results = []

    # -----------------------------------------------------------------
    # Density plot helper
    # -----------------------------------------------------------------
    def plot_density(f_true_g, f_hat_final, xg, p_val, alpha_val, rep_idx, save_path):
        plt.figure(figsize=(7, 4))
        plt.plot(xg, f_true_g, linewidth=2, label="True density")
        plt.plot(xg, f_hat_final, linewidth=2, label="Estimated density (KDE)")
        plt.xlabel("x")
        plt.ylabel("density")
        plt.title(rf"Estimated density: p={p_val}, $\alpha$={alpha_val:.2f}, rep={rep_idx:03d}")
        plt.grid(True, axis="y", ls="--", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    # -----------------------------------------------------------------
    # Scaling plot helper with trend line
    # -----------------------------------------------------------------
    def plot_with_trend(
        M,
        values,
        metric_name,
        p_val,
        alpha_val,
        rep_idx=None,
        drop_first=10,
        save_path=None,
    ):
        M = np.asarray(M, dtype=float)
        values = np.asarray(values, dtype=float)
        valid = np.isfinite(M) & np.isfinite(values) & (M > 0) & (values > 0)

        log_M = np.full_like(M, np.nan, dtype=float)
        log_val = np.full_like(values, np.nan, dtype=float)
        log_M[valid] = np.log10(M[valid])
        log_val[valid] = np.log10(values[valid])

        plt.figure(figsize=(6, 4))
        plt.plot(log_M, log_val, "o-", alpha=0.5, markersize=3, label="Distr. Loss")

        fit_mask = valid.copy()
        fit_mask[:drop_first] = False
        if np.count_nonzero(fit_mask) > 5:
            x_fit = log_M[fit_mask]
            y_fit = log_val[fit_mask]
            slope, intercept = np.polyfit(x_fit, y_fit, 1)
            y_pred = slope * x_fit + intercept
            plt.plot(x_fit, y_pred, "r--", linewidth=2, label=f"Fit slope: {slope:.3f}")

        plt.xlabel(r"$\log_{10}$(M$_n$)")

        if metric_name == "W1":
            metric_name_tex = r"W_1"
            title = rf"${metric_name_tex}$ Scaling: p={p_val}, $\alpha$={alpha_val:.2f}"
            ylabel = rf"$\log_{{10}}({metric_name_tex})$"
        elif metric_name == "MMD":
            metric_name_tex = r"\mathrm{MMD}"
            title = rf"${metric_name_tex}$ Scaling: p={p_val}, $\alpha$={alpha_val:.2f}"
            ylabel = rf"$\log_{{10}}({metric_name_tex})$"
        else:
            metric_name_tex = metric_name
            title = rf"{metric_name_tex} Scaling: p={p_val}, $\alpha$={alpha_val:.2f}"
            ylabel = rf"$\log_{{10}}$({metric_name_tex})"

        if rep_idx is not None:
            title = title + rf", rep={rep_idx:03d}"

        plt.title(title)
        plt.ylabel(ylabel)

        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.3)
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()
    
    


    # -----------------------------------------------------------------
    # Summary plot: alpha vs mean slope
    # -----------------------------------------------------------------
    USE_ERROR_BARS = False

    def plot_alpha_vs_mean_slope(
        metric_key,
        slopes_key,
        metric_label,
        filename=None,
        use_error_bars=USE_ERROR_BARS,
        ci_mult=1.96,
    ):
        plt.figure(figsize=(6, 4))
        DARK_BLUE = "#244B99"
        RED = "#CC0000"

        if metric_label == "W1":
            metric_label_tex = r"W_1"
        elif metric_label == "MMD":
            metric_label_tex = r"\mathrm{MMD}"
        else:
            metric_label_tex = metric_label

        unique_ps = sorted(set(res["p_target"] for res in all_results))

        for p in unique_ps:
            subset = [r for r in all_results if r["p_target"] == p]
            alpha_plot = np.array([r["alpha_real"] for r in subset], dtype=float)
            means = np.array([r[metric_key] for r in subset], dtype=float)

            order = np.argsort(alpha_plot)
            alpha_plot = alpha_plot[order]
            means = means[order]

            vals_plot = -means  # your sign convention

            if use_error_bars:
                stds = np.array([np.array(r[slopes_key]).std(ddof=1) for r in subset], dtype=float)
                stds = stds[order]
                err_plot = stds * (ci_mult / np.sqrt(n_reps))

                plt.errorbar(
                    alpha_plot,
                    vals_plot,
                    yerr=err_plot,
                    fmt="o-",
                    color=DARK_BLUE,
                    ecolor=DARK_BLUE,
                    elinewidth=1,
                    capsize=3,
                    label="Empirical",
                )
            else:
                plt.plot(alpha_plot, vals_plot, "o-", color=DARK_BLUE, label="Empirical")

            theo_vals = np.minimum(p, alpha_plot)
            plt.plot(alpha_plot, theo_vals, "--", color=RED, label="Theoretical")

        plt.xlabel(r"$\alpha$")
        plt.ylabel(rf"Mean Slope ${metric_label_tex}$")
        plt.title(rf"Mean ${metric_label_tex}$ Slope vs $\alpha$")
        plt.grid(True, axis="y", ls="--", alpha=0.3)

        plt.ylim(0.0, 0.65)
        plt.yticks(np.arange(0.0, 0.7, 0.1))

        handles, labels = plt.gca().get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        plt.legend(uniq.values(), uniq.keys())
        plt.tight_layout()

        if filename is not None:
            plt.savefig(fig_dir / filename, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    # -----------------------------------------------------------------
    # Main sweeps
    # -----------------------------------------------------------------
    for a_idx, alpha_real in enumerate(alphas):
        T = int(Ts[a_idx])
        DROP_FIRST_THIS = int(DROP_FIRST_FOR_FIT[a_idx])

        log("\n" + "#" * 80)
        log(f"=== Running all p for alphaReal = {alpha_real:.2f} ===")
        log(f"    T={T}")
        log(f"    drop_first_for_fit={DROP_FIRST_THIS}")
        log(f"    (W1 metric = exact sorted 1D empirical W1; w1_compare={W1_COMPARE})")
        log(f"    (MMD metric = sample-based unbiased MMD^2 -> sqrt)")
        log(f"    (KDE runs ONCE per repetition, for plotting only)")

        for p in p_values:
            log("\n" + "=" * 70)
            log(f"Running recursive NN estimator {n_reps}×: p_target={p}, alphaReal={alpha_real:.2f}")

            slopes_W1 = []
            slopes_MMD = []
            last_out = None

            for rep in range(n_reps):
                log(f"\n--- Repetition {rep + 1}/{n_reps} ---")
                rep_seed = BASE_SEED + 1000 * rep + int(1e6 * alpha_real) + int(1e3 * p)
                set_all_seeds(rep_seed, deterministic_torch=False)

                do_save_dens = (REP_DENS_MODE == "all") or (REP_DENS_MODE == "first" and rep == 0)

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
                    w1_mc_samples=W1_MC,
                    w1_compare=W1_COMPARE,
                    mmd_sigma=MMD_SIGMA,
                    mmd_mc_subsample=MMD_MC_SUBSAMPLE,
                    do_kde_for_plot=do_save_dens,  # KDE only if we will plot/save it
                    kde_plot_samples=KDE_PLOT_SAMPLES,
                    kde_bw=KDE_BW,
                    kde_clip=KDE_CLIP,
                    seed=rep_seed,
                )

                M_vals = out["M"]
                W1_vals = out["W1"]
                MMD_vals = out["MMD"]
                f_hist = out.get("f_hat_history", None)

                # Per-rep density plot saving (remake-compatible: f_hat_history[-1])
                if do_save_dens and f_hist is not None:
                    f_hat_final = f_hist[-1]
                    dens_path = fig_dir / f"density_rep={rep:03d}_p={p:.2f}_alphaReal={alpha_real:.2f}_T={T}.png"
                    plot_density(f_true_g, f_hat_final, xg, p, alpha_real, rep, dens_path)

                # Optional log correction when p ≈ alphaReal (kept convention)
                if np.isclose(p, alpha_real, atol=1e-8):
                    logM = np.log(np.maximum(M_vals, 2.0))
                    W1_for_slope = W1_vals / (logM)
                    MMD_for_slope = MMD_vals / (logM)
                else:
                    W1_for_slope = W1_vals
                    MMD_for_slope = MMD_vals

                drop_first = DROP_FIRST_THIS

                slope_W1, _ = estimate_power_law_slope(M_vals, W1_for_slope, drop_first=drop_first)
                slope_MMD, _ = estimate_power_law_slope(M_vals, MMD_for_slope, drop_first=drop_first)

                log(f"Rep {rep + 1}: slope_W1={slope_W1:.3f}, slope_MMD={slope_MMD:.3f}")

                slopes_W1.append(slope_W1)
                slopes_MMD.append(slope_MMD)
                last_out = out

                # Per-rep scaling plots (W1 + MMD)
                do_save_rep_fig = (REP_FIG_MODE == "all") or (REP_FIG_MODE == "first" and rep == 0)
                if do_save_rep_fig:
                    w1_rep_path = fig_dir / f"W1_rep={rep:03d}_p={p:.2f}_alphaReal={alpha_real:.2f}_T={T}.png"
                    plot_with_trend(
                        M_vals,
                        W1_for_slope,
                        "W1",
                        p,
                        alpha_real,
                        rep_idx=rep,
                        drop_first=drop_first,
                        save_path=w1_rep_path,
                    )

                    mmd_rep_path = fig_dir / f"MMD_rep={rep:03d}_p={p:.2f}_alphaReal={alpha_real:.2f}_T={T}.png"
                    plot_with_trend(
                        M_vals,
                        MMD_for_slope,
                        "MMD",
                        p,
                        alpha_real,
                        rep_idx=rep,
                        drop_first=drop_first,
                        save_path=mmd_rep_path,
                    )

            slopes_W1 = np.array(slopes_W1, dtype=float)
            slopes_MMD = np.array(slopes_MMD, dtype=float)

            if n_reps > 1:
                mean_W1, std_W1 = float(slopes_W1.mean()), float(slopes_W1.std(ddof=1))
                mean_MMD, std_MMD = float(slopes_MMD.mean()), float(slopes_MMD.std(ddof=1))
            else:
                mean_W1, std_W1 = float(slopes_W1.mean()), 0.0
                mean_MMD, std_MMD = float(slopes_MMD.mean()), 0.0

            theo_exp = min(p, alpha_real)

            log(f"\n*** Averaged slopes over {n_reps} runs ***")
            log(f"Mean slope W1   ~ M^{mean_W1:.3f}   (std={std_W1:.3f})")
            log(f"Mean slope MMD  ~ M^{mean_MMD:.3f}  (std={std_MMD:.3f})")
            log(f"Theoretical exponent (up to logs): min(p, alphaReal) = {theo_exp:.3f}")
            if np.isclose(p, alpha_real, atol=1e-8):
                log("Note: W1 divided by log(M_n), and MMD divided by log(M_n) before slope fitting (p ≈ alphaReal).")

            all_results.append(
                {
                    **last_out,
                    "p_target": p,
                    "alpha_real": float(alpha_real),
                    "alpha_synth": float(1.0 - alpha_real),
                    "T": int(T),
                    "drop_first_for_fit": int(drop_first),
                    "slopes_W1": slopes_W1,
                    "slopes_MMD": slopes_MMD,
                    "mean_slope_W1": float(mean_W1),
                    "mean_slope_MMD": float(mean_MMD),
                }
            )

    # Summary plots
    plot_alpha_vs_mean_slope(
        metric_key="mean_slope_W1",
        slopes_key="slopes_W1",
        metric_label="W1",
        filename="alphaReal_vs_mean_slope_W1.png",
    )

    plot_alpha_vs_mean_slope(
        metric_key="mean_slope_MMD",
        slopes_key="slopes_MMD",
        metric_label="MMD",
        filename="alphaReal_vs_mean_slope_MMD.png",
    )

    # Save all_results as JSON
    results_path = out_dir / "all_results.json"
    with open(results_path, "w") as f:
        json.dump(
            [{k: _make_json_serializable(v) for k, v in res.items()} for res in all_results],
            f,
            indent=2,
        )
    log(f"Saved results JSON to {results_path}")

    # Save summary CSV
    summary_path = out_dir / "summary_slopes.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "p_target",
                "alpha_real",
                "alpha_synth",
                "T",
                "drop_first_for_fit",
                "m_total",
                "m1",
                "m2",
                "mean_slope_W1",
                "mean_slope_MMD",
                "model_type",
                "MMD_sigma",
                "MMD_mc_subsample",
                "KDE_plot_samples",
                "KDE_bw",
            ]
        )
        for res in all_results:
            writer.writerow(
                [
                    res["p_target"],
                    res["alpha_real"],
                    res["alpha_synth"],
                    res.get("T", ""),
                    res.get("drop_first_for_fit", ""),
                    res["m_total"],
                    res["m1"],
                    res["m2"],
                    res["mean_slope_W1"],
                    res["mean_slope_MMD"],
                    res.get("model_type", "mlp"),
                    res.get("mmd_sigma", MMD_SIGMA),
                    res.get("mmd_mc_subsample", MMD_MC_SUBSAMPLE),
                    res.get("kde_plot_samples", KDE_PLOT_SAMPLES),
                    res.get("kde_bw", KDE_BW),
                ]
            )
    log(f"Saved summary CSV to {summary_path}")
