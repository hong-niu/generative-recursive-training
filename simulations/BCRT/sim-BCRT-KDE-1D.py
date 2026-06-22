#!/usr/bin/env python3
"""
Recursive density estimation under real–synthetic mixing (biased-real stream; ECDF option).

This script runs a recursive estimator over T outer iterations. At each iteration n:

  (1) Draw m1 NEW real samples from a *biased* distribution P_n.
  (2) Draw m2 synthetic samples from the *current* estimator and append them to
      the synthetic pool (memory).
  (3) Recompute the estimator from the accumulated dataset.

Outputs:
  - Per-repetition compressed .npz files containing M, W1, MMD, MMD2, bandwidths,
    slope-fitting arrays, grid/truth arrays, and optionally full f/F histories.
  - Per-repetition metric CSV files, useful for remaking slope/rate plots without loading .npz.
  - Density plots organized by alpha/q/p combination under figures/density_plots/.
  - Rate plots organized by alpha/q/p combination under figures/rate_plots/.
  - summary_table.csv: theoretical rate vs observed W1/MMD rate by alpha/q/p combination.
  - all_rep_index.csv/json: file index for every repetition.
  - all_results.json: combination-level summaries with per-repetition file paths.

Notes:
  - MMD is computed as sqrt(MMD^2) using Gaussian-kernel grid quadrature.
  - Negative or zero MMD^2 values are treated as invalid for MMD/slope fitting.
"""


import os
import json
import csv
import datetime
import logging
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm
from scipy.special import ndtr
from sklearn.linear_model import LinearRegression

import torch


# ============================================================
# 1. Unbiased target (2-Gaussian mixture) + biased sampler (3rd component)
# ============================================================
w1 = 0.35
mu1, sigma1 = -2.0, 0.8
mu2, sigma2 =  1.0, 1.3

# Biased extra component 
mu3, sigma3 = 3.0, 1.0


def sample_true_np(n, rng, bias=0.0):
    if n <= 0:
        return np.empty(0, float)
    bias = float(np.clip(bias, 0.0, 1.0))

    u = rng.random(n)
    m3 = u < bias

    u2 = rng.random(n)
    m1_mask = (~m3) & (u2 < w1)

    n3 = int(m3.sum())
    n1 = int(m1_mask.sum())
    n2 = n - n1 - n3

    x = np.empty(n, float)
    i = 0
    if n3:
        x[i:i+n3] = rng.normal(mu3, sigma3, n3)
        i += n3
    if n1:
        x[i:i+n1] = rng.normal(mu1, sigma1, n1)
        i += n1
    if n2:
        x[i:i+n2] = rng.normal(mu2, sigma2, n2)
    return x


def true_pdf_np(x):
    return w1 * norm.pdf(x, mu1, sigma1) + (1 - w1) * norm.pdf(x, mu2, sigma2)


def true_cdf_np(x):
    return w1 * ndtr((x - mu1) / sigma1) + (1 - w1) * ndtr((x - mu2) / sigma2)


def true_pdf_torch(x_t):
    c1 = w1 / (sigma1 * np.sqrt(2 * np.pi))
    c2 = (1 - w1) / (sigma2 * np.sqrt(2 * np.pi))
    z1 = (x_t - mu1) / sigma1
    z2 = (x_t - mu2) / sigma2
    return c1 * torch.exp(-0.5 * z1**2) + c2 * torch.exp(-0.5 * z2**2)


def true_cdf_torch(x_t):
    z1 = (x_t - mu1) / sigma1
    z2 = (x_t - mu2) / sigma2
    Phi1 = 0.5 * (1 + torch.erf(z1 / np.sqrt(2.0)))
    Phi2 = 0.5 * (1 + torch.erf(z2 / np.sqrt(2.0)))
    return w1 * Phi1 + (1 - w1) * Phi2


def sample_true_torch(n, device, dtype=torch.float64, bias=0.0):
    if n <= 0:
        return torch.empty(0, device=device, dtype=dtype)
    bias = float(np.clip(bias, 0.0, 1.0))

    u = torch.rand(n, device=device, dtype=dtype)
    m3 = u < bias

    u2 = torch.rand(n, device=device, dtype=dtype)
    m1_mask = (~m3) & (u2 < w1)

    n3 = int(m3.sum().item())
    n1 = int(m1_mask.sum().item())
    n2 = n - n1 - n3

    xs = []
    if n3:
        xs.append(torch.randn(n3, device=device, dtype=dtype) * sigma3 + mu3)
    if n1:
        xs.append(torch.randn(n1, device=device, dtype=dtype) * sigma1 + mu1)
    if n2:
        xs.append(torch.randn(n2, device=device, dtype=dtype) * sigma2 + mu2)

    return torch.cat(xs, dim=0) if xs else torch.empty(0, device=device, dtype=dtype)


left = min(mu1 - 6 * sigma1, mu2 - 6 * sigma2)
right = max(mu1 + 6 * sigma1, mu2 + 6 * sigma2)
m_grid = 500

xg = np.linspace(left, right, m_grid)
dx = xg[1] - xg[0]
F_true_g = true_cdf_np(xg)
f_true_g = true_pdf_np(xg)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64

xg_t = torch.from_numpy(xg).to(device=DEVICE, dtype=DTYPE)
dx_t = xg_t[1] - xg_t[0]

F_true_g_t = torch.from_numpy(F_true_g).to(device=DEVICE, dtype=DTYPE)
f_true_g_t = torch.from_numpy(f_true_g).to(device=DEVICE, dtype=DTYPE)

MMD_SIGMA = 0.5
Xgrid = xg_t.view(-1, 1)
K_mmd = torch.exp(-(Xgrid - Xgrid.T) ** 2 / (2.0 * MMD_SIGMA**2))


# ============================================================
# 2. Helper routines 
# ============================================================
def silverman_bw_np(x):
    s = np.std(x, ddof=1)
    n = len(x)
    if n <= 1 or s == 0:
        return 1.0
    return 1.06 * s * n ** (-1 / 5)


def cdf_from_pdf_on_grid_torch(pdf_vals, dx, renormalize=True):
    F = torch.empty_like(pdf_vals)
    F[0] = 0.0
    F[1:] = torch.cumsum((pdf_vals[:-1] + pdf_vals[1:]) * 0.5 * dx, dim=0)
    if renormalize:
        total = F[-1].clamp_min(1e-12)
        F = F / total
    return F


def w1_from_cdfs_on_grid_torch(F_true, F_model, xg_t):
    return torch.trapz(torch.abs(F_true - F_model), xg_t)


def empirical_cdf_on_grid_torch(samples, xg_t):
    s, _ = torch.sort(samples)
    n = s.shape[0]
    idx = torch.searchsorted(s, xg_t, right=True)
    return idx.to(dtype=DTYPE) / float(n)


def mmd2_from_pdfs_torch(f_true, f_model, K, dx):
    v_true = torch.mv(K, f_true)
    v_model = torch.mv(K, f_model)
    dx2 = dx * dx
    return (torch.dot(f_true, v_true) + torch.dot(f_model, v_model) - 2.0 * torch.dot(f_true, v_model)) * dx2


# ============================================================
# 3. Recursive estimator (biased real stream)
# ============================================================
def run_recursive_estimator(
    p_target=0.5,
    alpha=0.5,
    m1=50,
    T=200,
    use_ecdf=False,
    h0=0.5,
    bias_rate=0.0
):
    device, dtype = DEVICE, DTYPE
    m2 = 0 if alpha <= 0 else int(round(alpha * m1 / (1 - alpha)))
    alpha_used = m2 / (m1 + m2) if (m1 + m2) > 0 else 0.0

    X_real = torch.empty(0, device=device, dtype=dtype)
    X_synth = torch.empty(0, device=device, dtype=dtype)

    Ms = np.empty(T, int)
    W1s = np.empty(T, float)
    MMDs = np.empty(T, float)
    MMD2s = np.empty(T, float)
    bandwidths = np.empty(T, float)

    f_hat_history = []
    F_hat_history = []

    h0_ = None
    normal_const = 1.0 / np.sqrt(2.0 * np.pi)

    with torch.no_grad():
        for n in range(1, T + 1):
            idx = n - 1
            bias = (0.2) * (n + 5) ** (-bias_rate) if bias_rate > 0 else 0.0

            x_new_real = sample_true_torch(m1, device=device, dtype=dtype, bias=bias)
            X_real = torch.cat((X_real, x_new_real), dim=0)

            X_train = torch.cat((X_real, X_synth), dim=0) if X_synth.numel() > 0 else X_real
            M_n = X_train.shape[0]
            Ms[idx] = int(M_n)

            if use_ecdf:
                F_hat_g_t = empirical_cdf_on_grid_torch(X_train, xg_t)
                dx_vec = torch.full_like(F_hat_g_t, dx_t)
                f_hat_g_t = torch.gradient(F_hat_g_t, spacing=(dx_vec,))[0]
                bandwidths[idx] = 0.0

                if m2 > 0:
                    ir = torch.randint(0, M_n, (m2,), device=device)
                    X_synth = torch.cat((X_synth, X_train[ir]), dim=0)

            else:
                if h0_ is None:
                    h0_ = h0 if h0 > 0 else 0.5

                # h_n = h0_ * (n) ** (-p_target / 2.0)
                h_n = h0_ * (n) ** (-p_target)
                bandwidths[idx] = float(h_n)
                h_n_t = torch.tensor(h_n, device=device, dtype=dtype)

                diff = (xg_t.unsqueeze(1) - X_train.unsqueeze(0)) / h_n_t
                f_hat_g_t = torch.exp(-0.5 * diff**2).mean(dim=1) * (normal_const / h_n_t)
                F_hat_g_t = cdf_from_pdf_on_grid_torch(f_hat_g_t, dx_t, True)

                if m2 > 0:
                    ir = torch.randint(0, M_n, (m2,), device=device)
                    synth = X_train[ir] + torch.randn(m2, device=device, dtype=dtype) * h_n_t
                    X_synth = torch.cat((X_synth, synth), dim=0)

            W1s[idx] = float(w1_from_cdfs_on_grid_torch(F_true_g_t, F_hat_g_t, xg_t).cpu().item())
            
            mmd2_val = float(mmd2_from_pdfs_torch(f_true_g_t, f_hat_g_t, K_mmd, float(dx)).cpu().item())
            MMD2s[idx] = mmd2_val
            
            if np.isfinite(mmd2_val) and mmd2_val > 0:
                MMDs[idx] = float(np.sqrt(mmd2_val))
            else:
                MMDs[idx] = np.nan

            f_hat_history.append(f_hat_g_t.cpu().numpy())
            F_hat_history.append(F_hat_g_t.cpu().numpy())

            if (n == 1) or (n % 50 == 0):
                tag = "ECDF" if use_ecdf else "KDE"
                hn = 0.0 if use_ecdf else h_n
                print(
                    f"[{tag} p={p_target:.2f}, alpha≈{alpha_used:.2f}, q={bias_rate:.2f}] "
                    f"n={n:4d} M={M_n:6d} bias={bias:.4g} h={hn:.4g} W1={W1s[idx]:.6f} MMD={MMDs[idx]:.6f}"
                )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return dict(
        M=Ms,
        W1=W1s,
        MMD=MMDs,
        MMD2=MMD2s,
        bandwidths=bandwidths,
        p_target=p_target,
        alpha=alpha_used,
        m1=m1,
        m2=m2,
        use_ecdf=use_ecdf,
        f_hat_history=np.stack(f_hat_history),
        F_hat_history=np.stack(F_hat_history),
        bias_rate=bias_rate,
        # Requisite metadata to remake figures identically from JSON
        grid={"left": float(left), "right": float(right), "m_grid": int(m_grid)},
        true_mixture={
            "w1": float(w1),
            "mu1": float(mu1), "sigma1": float(sigma1),
            "mu2": float(mu2), "sigma2": float(sigma2),
            "mu3": float(mu3), "sigma3": float(sigma3)
        }
    )


# ============================================================
# 4. Power-law slope estimators + JSON helper 
# ============================================================
def estimate_power_law_slope(M, vals, drop_first=10):
    M = np.asarray(M, dtype=float)
    vals = np.asarray(vals, dtype=float)

    if drop_first >= len(M) - 5:
        drop_first = max(0, len(M) - 5)

    M_use = M[drop_first:]
    vals_use = vals[drop_first:]

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
    # Reproducibility: each repetition gets a deterministic seed derived from this.
    BASE_SEED = 12345

    p_values = [0.5]
    alphas = [0.25, 0.5, 0.75]
    bias_rates = [0.25, 0.5, 0.75]

    T, m1, n_reps, h0, drop_first = 3000, 50, 100, 2, 200 

    # Saving controls.
    # Full histories are enough to remake density curves at every iteration, but can be large:
    SAVE_FULL_HISTORY = True
    SAVE_REP_METRICS_CSV = True
    REP_DENS_MODE = "all"      # "all" or "first"
    REP_RATE_MODE = "all"      # "all" or "first"

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"biased_alphaReal_h0={h0}_T={T}_m1={m1}_nreps={n_reps}_{ts}"
    out_dir = Path("logs") / "KDE" / run_name
    fig_dir = out_dir / "figures"
    density_root = fig_dir / "density_plots"
    rate_root = fig_dir / "rate_plots"
    data_root = out_dir / "rep_data"

    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    density_root.mkdir(parents=True, exist_ok=True)
    rate_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)

    log_file = out_dir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    log = logging.info

    def set_rep_seed(seed: int):
        """Seed numpy + torch for reproducible rep-level runs."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def combo_label(p_val, alpha_real_val, bias_rate):
        return f"p={p_val:.2f}_alphaReal={alpha_real_val:.2f}_q={bias_rate:.2f}"

    def combo_dirname(p_val, alpha_real_val, bias_rate):
        return combo_label(p_val, alpha_real_val, bias_rate).replace("=", "").replace(".", "p")

    def log_config(title: str, cfg: dict):
        log("\n" + "-" * 80)
        log(title)
        log(json.dumps(cfg, indent=2, sort_keys=True))
        log("-" * 80 + "\n")

    RUN_CONFIG = {
        "base_seed": BASE_SEED,
        "p_values": p_values,
        "alphas_real": alphas,
        "bias_rates": bias_rates,
        "T": T,
        "m1": m1,
        "n_reps": n_reps,
        "h0": h0,
        "drop_first_for_fit": drop_first,
        "mmd_sigma": MMD_SIGMA,
        "save_full_history": SAVE_FULL_HISTORY,
        "save_rep_metrics_csv": SAVE_REP_METRICS_CSV,
        "rep_density_mode": REP_DENS_MODE,
        "rep_rate_mode": REP_RATE_MODE,
        "output_layout": {
            "rep_npz_data": str(data_root),
            "rep_density_plots": str(density_root),
            "rep_rate_plots": str(rate_root),
            "summary_table": str(out_dir / "summary_table.csv"),
            "rep_index": str(out_dir / "all_rep_index.csv"),
        },
        "grid": {"left": float(left), "right": float(right), "m_grid": int(m_grid)},
        "true_mixture": {
            "w1": float(w1),
            "mu1": float(mu1), "sigma1": float(sigma1),
            "mu2": float(mu2), "sigma2": float(sigma2),
            "mu3": float(mu3), "sigma3": float(sigma3),
        },
    }
    log_config("RUN CONFIG", RUN_CONFIG)

    def plot_density_rep(f_true_g, f_hat_t, xg, p_val, alpha_real, bias_rate, rep_idx, save_path):
        plt.figure(figsize=(7, 4))
        plt.plot(xg, f_true_g, linewidth=2, label="True density")
        plt.plot(xg, f_hat_t, linewidth=2, label="Estimated density")
        plt.xlabel("x")
        plt.ylabel("density")
        plt.title(
            f"Estimated density (final): p={p_val}, alpha_real={alpha_real:.2f}, "
            f"q={bias_rate:.2f}, rep={rep_idx:03d}"
        )
        plt.grid(True, axis="y", ls="--", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def plot_with_trend(
        M,
        values,
        metric_name,
        p_val,
        alpha_real_val,
        bias_rate,
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
        elif metric_name == "MMD":
            metric_name_tex = r"\mathrm{MMD}"
        else:
            metric_name_tex = metric_name

        plt.ylabel(rf"$\log_{{10}}({metric_name_tex})$")
        title = rf"${metric_name_tex}$ Scaling: p={p_val}, $\alpha_{{real}}$={alpha_real_val:.2f}, q={bias_rate:.2f}"
        if rep_idx is not None:
            title += rf", rep={rep_idx:03d}"
        plt.title(title)

        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.3)
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def save_rep_outputs(
        out,
        M_vals,
        W1_for_slope,
        MMD_for_slope,
        p_val,
        alpha_real_val,
        alpha_est_val,
        bias_rate,
        rep_idx,
        rep_seed,
        slope_W1,
        slope_MMD,
        intercept_W1,
        intercept_MMD,
        data_dir,
    ):
        """Save every repetition's data in remake-friendly files."""
        rep_tag = f"rep={rep_idx:03d}"
        rep_meta = {
            "rep": int(rep_idx),
            "rep_seed": int(rep_seed),
            "p_target": float(p_val),
            "alpha_real": float(alpha_real_val),
            "alpha_est": float(alpha_est_val),
            "bias_rate": float(bias_rate),
            "T": int(len(M_vals)),
            "m1": int(out["m1"]),
            "m2": int(out["m2"]),
            "h0": float(h0),
            "drop_first_for_fit": int(drop_first),
            "mmd_sigma": float(MMD_SIGMA),
            "slope_W1": float(slope_W1),
            "slope_MMD": float(slope_MMD),
            "intercept_W1": float(intercept_W1),
            "intercept_MMD": float(intercept_MMD),
            "observed_rate_W1": float(-slope_W1),
            "observed_rate_MMD": float(-slope_MMD),
            "theoretical_rate": float(min(p_val, alpha_real_val, bias_rate)),
            "grid": {"left": float(left), "right": float(right), "m_grid": int(m_grid)},
        }

        npz_path = data_dir / f"{rep_tag}_data.npz"
        npz_payload = {
            "M": np.asarray(out["M"]),
            "W1": np.asarray(out["W1"]),
            "MMD": np.asarray(out["MMD"]),
            "MMD2": np.asarray(out["MMD2"]),
            "bandwidths": np.asarray(out["bandwidths"]),
            "W1_for_slope": np.asarray(W1_for_slope),
            "MMD_for_slope": np.asarray(MMD_for_slope),
            "xg": np.asarray(xg),
            "f_true_g": np.asarray(f_true_g),
            "F_true_g": np.asarray(F_true_g),
            "metadata_json": np.array(json.dumps(rep_meta, sort_keys=True)),
        }
        if SAVE_FULL_HISTORY:
            npz_payload["f_hat_history"] = np.asarray(out["f_hat_history"])
            npz_payload["F_hat_history"] = np.asarray(out["F_hat_history"])
        else:
            # Final curves are enough to remake final density plots while keeping files smaller.
            npz_payload["f_hat_final"] = np.asarray(out["f_hat_history"][-1])
            npz_payload["F_hat_final"] = np.asarray(out["F_hat_history"][-1])

        np.savez_compressed(npz_path, **npz_payload)

        json_path = data_dir / f"{rep_tag}_summary.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(rep_meta, f, indent=2, sort_keys=True)

        metrics_csv_path = None
        if SAVE_REP_METRICS_CSV:
            metrics_csv_path = data_dir / f"{rep_tag}_metrics.csv"
            with open(metrics_csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["iter", "M", "W1", "MMD", "MMD2", "bandwidth", "W1_for_slope", "MMD_for_slope"])
                for i in range(len(M_vals)):
                    w.writerow([
                        i + 1,
                        int(out["M"][i]),
                        float(out["W1"][i]),
                        float(out["MMD"][i]) if np.isfinite(out["MMD"][i]) else "nan",
                        float(out["MMD2"][i]) if np.isfinite(out["MMD2"][i]) else "nan",
                        float(out["bandwidths"][i]),
                        float(W1_for_slope[i]) if np.isfinite(W1_for_slope[i]) else "nan",
                        float(MMD_for_slope[i]) if np.isfinite(MMD_for_slope[i]) else "nan",
                    ])

        return {
            **rep_meta,
            "npz_path": str(npz_path),
            "summary_json_path": str(json_path),
            "metrics_csv_path": str(metrics_csv_path) if metrics_csv_path is not None else "",
        }

    # Each entry is one repetition, not just one alpha/q/p combination.
    rep_index = []
    # Each entry is one alpha/q/p combination with arrays of rep slopes and paths.
    all_results = []

    for alpha_real in alphas:
        # run_recursive_estimator's alpha argument is the synthetic fraction.
        alpha_est = 1.0 - alpha_real

        for q in bias_rates:
            for p in p_values:
                label = combo_label(p, alpha_real, q)
                dirname = combo_dirname(p, alpha_real, q)
                dens_dir = density_root / dirname
                rate_dir = rate_root / dirname
                data_dir = data_root / dirname
                dens_dir.mkdir(parents=True, exist_ok=True)
                rate_dir.mkdir(parents=True, exist_ok=True)
                data_dir.mkdir(parents=True, exist_ok=True)

                log("\n" + "#" * 80)
                log(f"=== {label} (alpha_est={alpha_est:.2f}) ===")
                log("\n" + "=" * 70)
                log(f"Running {n_reps}×: p={p}, alpha_real={alpha_real:.2f}, q={q:.2f}")

                slopes_W1 = []
                slopes_MMD = []
                rep_paths = []
                alpha_real_used_final = None
                alpha_est_used_final = None

                for rep in range(n_reps):
                    rep_seed = BASE_SEED + 1_000_000 * int(round(100 * alpha_real)) + 10_000 * int(round(100 * q)) + 100 * int(round(100 * p)) + rep
                    set_rep_seed(rep_seed)

                    log(f"--- rep {rep+1}/{n_reps} | seed={rep_seed} ---")
                    out = run_recursive_estimator(
                        p_target=p,
                        alpha=alpha_est,
                        m1=m1,
                        T=T,
                        use_ecdf=False,
                        h0=h0,
                        bias_rate=q,
                    )

                    M_vals = out["M"]
                    W1_vals = out["W1"]
                    MMD_vals = out["MMD"]

                    alpha_est_used = float(out["alpha"])
                    alpha_real_used = 1.0 - alpha_est_used
                    alpha_real_used_final = alpha_real_used
                    alpha_est_used_final = alpha_est_used

                    if np.isclose(p, alpha_real_used, atol=1e-8):
                        logM = np.log(np.maximum(M_vals, 2.0))
                        W1_for_slope = W1_vals / logM
                        MMD_for_slope = MMD_vals / logM
                    else:
                        W1_for_slope = W1_vals
                        MMD_for_slope = MMD_vals

                    sW1, bW1 = estimate_power_law_slope(M_vals, W1_for_slope, drop_first=drop_first)
                    sMMD, bMMD = estimate_power_law_slope(M_vals, MMD_for_slope, drop_first=drop_first)
                    slopes_W1.append(sW1)
                    slopes_MMD.append(sMMD)

                    log(f"rep {rep+1}: slope_W1={sW1:.3f} slope_MMD={sMMD:.3f}")

                    do_save_dens = (REP_DENS_MODE == "all") or (REP_DENS_MODE == "first" and rep == 0)
                    if do_save_dens:
                        dens_path = dens_dir / f"density_rep={rep:03d}_{label}.png"
                        plot_density_rep(f_true_g, out["f_hat_history"][-1], xg, p, alpha_real_used, q, rep, dens_path)
                        # log(f"Saved density plot -> {dens_path}")
                    else:
                        dens_path = None

                    do_save_rate = (REP_RATE_MODE == "all") or (REP_RATE_MODE == "first" and rep == 0)
                    if do_save_rate:
                        w1_rate_path = rate_dir / f"W1_rep={rep:03d}_{label}.png"
                        mmd_rate_path = rate_dir / f"MMD_rep={rep:03d}_{label}.png"
                        plot_with_trend(
                            M_vals,
                            W1_for_slope,
                            "W1",
                            p,
                            alpha_real_used,
                            q,
                            rep_idx=rep,
                            drop_first=drop_first,
                            save_path=w1_rate_path,
                        )
                        plot_with_trend(
                            M_vals,
                            MMD_for_slope,
                            "MMD",
                            p,
                            alpha_real_used,
                            q,
                            rep_idx=rep,
                            drop_first=drop_first,
                            save_path=mmd_rate_path,
                        )
                        # log(f"Saved rate plots -> {w1_rate_path}, {mmd_rate_path}")
                    else:
                        w1_rate_path = None
                        mmd_rate_path = None

                    rep_record = save_rep_outputs(
                        out=out,
                        M_vals=M_vals,
                        W1_for_slope=W1_for_slope,
                        MMD_for_slope=MMD_for_slope,
                        p_val=p,
                        alpha_real_val=alpha_real_used,
                        alpha_est_val=alpha_est_used,
                        bias_rate=q,
                        rep_idx=rep,
                        rep_seed=rep_seed,
                        slope_W1=sW1,
                        slope_MMD=sMMD,
                        intercept_W1=bW1,
                        intercept_MMD=bMMD,
                        data_dir=data_dir,
                    )
                    rep_record.update({
                        "combo_label": label,
                        "density_plot_path": str(dens_path) if dens_path is not None else "",
                        "W1_rate_plot_path": str(w1_rate_path) if w1_rate_path is not None else "",
                        "MMD_rate_plot_path": str(mmd_rate_path) if mmd_rate_path is not None else "",
                    })
                    rep_index.append(rep_record)
                    rep_paths.append(rep_record)

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                slopes_W1 = np.array(slopes_W1, dtype=float)
                slopes_MMD = np.array(slopes_MMD, dtype=float)

                mean_W1 = float(np.nanmean(slopes_W1))
                mean_MMD = float(np.nanmean(slopes_MMD))
                std_W1 = float(np.nanstd(slopes_W1, ddof=1)) if n_reps > 1 else 0.0
                std_MMD = float(np.nanstd(slopes_MMD, ddof=1)) if n_reps > 1 else 0.0
                median_W1 = float(np.nanmedian(slopes_W1))
                median_MMD = float(np.nanmedian(slopes_MMD))
                theo = float(min(p, alpha_real_used_final, q))

                log(f"*** avg over {n_reps} ***")
                log(f"mean slope W1={mean_W1:.3f} (std={std_W1:.3f}) | observed rate≈{-mean_W1:.3f}")
                log(f"mean slope MMD={mean_MMD:.3f} (std={std_MMD:.3f}) | observed rate≈{-mean_MMD:.3f}")
                log(f"theory rate min(p,alpha_real,q)={theo:.3f}")

                all_results.append({
                    "combo_label": label,
                    "p_target": float(p),
                    "alpha_real": float(alpha_real_used_final),
                    "alpha_est": float(alpha_est_used_final),
                    "bias_rate": float(q),
                    "T": int(T),
                    "m1": int(m1),
                    "m2": int(rep_paths[-1]["m2"]),
                    "drop_first_for_fit": int(drop_first),
                    "theoretical_rate": theo,
                    "slopes_W1": slopes_W1,
                    "slopes_MMD": slopes_MMD,
                    "mean_slope_W1": mean_W1,
                    "std_slope_W1": std_W1,
                    "median_slope_W1": median_W1,
                    "mean_slope_MMD": mean_MMD,
                    "std_slope_MMD": std_MMD,
                    "median_slope_MMD": median_MMD,
                    "observed_rate_mean_W1": float(-mean_W1),
                    "observed_rate_median_W1": float(-median_W1),
                    "observed_rate_mean_MMD": float(-mean_MMD),
                    "observed_rate_median_MMD": float(-median_MMD),
                    "rep_data": rep_paths,
                })

    # -----------------------------------------------------------------
    # End-of-run summaries across alpha/q/p combinations
    # -----------------------------------------------------------------
    summary = []
    for r in all_results:
        summary.append(dict(
            alpha_real=r["alpha_real"],
            alpha_est=r["alpha_est"],
            bias_rate=r["bias_rate"],
            p_target=r["p_target"],
            T=r["T"],
            m1=r["m1"],
            m2=r["m2"],
            n_reps=n_reps,
            theoretical_rate=r["theoretical_rate"],
            observed_rate_mean_W1=r["observed_rate_mean_W1"],
            observed_rate_median_W1=r["observed_rate_median_W1"],
            observed_rate_std_W1=float(np.nanstd(-np.asarray(r["slopes_W1"], dtype=float), ddof=1)) if n_reps > 1 else 0.0,
            observed_rate_mean_MMD=r["observed_rate_mean_MMD"],
            observed_rate_median_MMD=r["observed_rate_median_MMD"],
            observed_rate_std_MMD=float(np.nanstd(-np.asarray(r["slopes_MMD"], dtype=float), ddof=1)) if n_reps > 1 else 0.0,
        ))

    print("\nSUMMARY: observed rate = -slope")
    print(
        f"{'alpha':>6} {'q':>6} {'p':>5} {'theory':>10} "
        f"{'obs_W1_mean':>12} {'obs_W1_med':>11} {'obs_MMD_mean':>13} {'obs_MMD_med':>12}"
    )
    for s in summary:
        print(
            f"{s['alpha_real']:6.2f} {s['bias_rate']:6.2f} {s['p_target']:5.2f} "
            f"{s['theoretical_rate']:10.3f} {s['observed_rate_mean_W1']:12.3f} "
            f"{s['observed_rate_median_W1']:11.3f} {s['observed_rate_mean_MMD']:13.3f} "
            f"{s['observed_rate_median_MMD']:12.3f}"
        )

    summary_path = out_dir / "summary_table.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "alpha_real", "alpha_est", "bias_rate", "p_target", "T", "m1", "m2", "n_reps",
            "theoretical_rate",
            "observed_rate_mean_W1", "observed_rate_median_W1", "observed_rate_std_W1",
            "observed_rate_mean_MMD", "observed_rate_median_MMD", "observed_rate_std_MMD",
        ])
        for s in summary:
            w.writerow([
                s["alpha_real"], s["alpha_est"], s["bias_rate"], s["p_target"],
                s["T"], s["m1"], s["m2"], s["n_reps"], s["theoretical_rate"],
                s["observed_rate_mean_W1"], s["observed_rate_median_W1"], s["observed_rate_std_W1"],
                s["observed_rate_mean_MMD"], s["observed_rate_median_MMD"], s["observed_rate_std_MMD"],
            ])
    log(f"Saved summary table -> {summary_path}")

    rep_index_path = out_dir / "all_rep_index.csv"
    with open(rep_index_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "combo_label", "rep", "rep_seed", "p_target", "alpha_real", "alpha_est", "bias_rate",
            "T", "m1", "m2", "h0", "drop_first_for_fit", "mmd_sigma",
            "slope_W1", "slope_MMD", "observed_rate_W1", "observed_rate_MMD", "theoretical_rate",
            "npz_path", "summary_json_path", "metrics_csv_path",
            "density_plot_path", "W1_rate_plot_path", "MMD_rate_plot_path",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rep_index:
            w.writerow(row)
    log(f"Saved per-repetition index -> {rep_index_path}")

    with open(out_dir / "all_results.json", "w", encoding="utf-8") as f:
        json.dump([{k: _make_json_serializable(v) for k, v in r.items()} for r in all_results], f, indent=2)
    log(f"Saved combination-level JSON to {out_dir / 'all_results.json'}")

    with open(out_dir / "all_rep_index.json", "w", encoding="utf-8") as f:
        json.dump(rep_index, f, indent=2)
    log(f"Saved per-repetition JSON index to {out_dir / 'all_rep_index.json'}")
