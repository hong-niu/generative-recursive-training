#!/usr/bin/env python3
"""
Recursive density estimation under real–synthetic mixing (biased-real stream; ECDF option).

This script runs a recursive estimator over T outer iterations. At each iteration n:

  (1) Draw m1 NEW real samples from a *biased* distribution P_n.
  (2) Draw m2 synthetic samples from the *current* estimator and append them to
      the synthetic pool (memory).
  (3) Recompute the estimator from the accumulated dataset.
  (4) Evaluate convergence to the fixed target distribution P0 (unbiased mixture).

Outputs:
  - Per-repetition compressed .npz files containing M, W1, MMD, MMD2, bandwidths,
    biases, slope-fitting arrays, grid/truth arrays, and optionally full f/F histories.
  - Per-repetition metric CSV files, useful for remaking slope/rate plots without loading .npz.
  - Density plots organized by alpha/q/p combination under figures/density_plots/.
  - Rate plots organized by alpha/q/p combination under figures/rate_plots/.
  - summary_table.csv: theoretical rate vs observed W1/MMD rate by alpha/q/p combination.
  - all_rep_index.csv/json: file index for every repetition.
  - all_results.json: combination-level summaries with per-repetition file paths.

Notes: 
    - MMD is computed as the Gaussian-kernel grid-quadrature MMD, using sqrt(MMD^2).
    - If numerical quadrature gives a non-positive MMD^2 value, that point is stored as NaN
    and excluded from the log–log slope fit.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import json
import csv
import datetime
import logging
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.neighbors import KernelDensity
from scipy.stats import norm
from scipy.special import ndtr
from sklearn.linear_model import LinearRegression

# ============================================================
# 1. Target distribution P0 (fixed two-component Gaussian mixture)
# ============================================================
w1 = 0.35
mu1, sigma1 = -2.0, 0.8
mu2, sigma2 = 1.0, 1.3

# Bias component Q 
mu3, sigma3 = 2.0, 1.0

def sample_true(n, rng, bias=0.0):
    if n <= 0: return np.empty(0)
    u = rng.random(n)
    x = np.empty(n)
    x[u < bias] = rng.normal(mu3, sigma3, (u < bias).sum())
    x[(u >= bias) & (u < bias + (1-bias)*w1)] = rng.normal(mu1, sigma1, ((u >= bias) & (u < bias + (1-bias)*w1)).sum())
    x[u >= bias + (1-bias)*w1] = rng.normal(mu2, sigma2, (u >= bias + (1-bias)*w1).sum())
    return x


def true_pdf(x):
    return w1 * norm.pdf(x, mu1, sigma1) + (1 - w1) * norm.pdf(x, mu2, sigma2)


def true_cdf(x):
    return w1 * ndtr((x - mu1) / sigma1) + (1 - w1) * ndtr((x - mu2) / sigma2)


left = min(mu1 - 6 * sigma1, mu2 - 6 * sigma2)
right = max(mu1 + 6 * sigma1, mu2 + 6 * sigma2)
m_grid = 200  
xg = np.linspace(left, right, m_grid)
dx = xg[1] - xg[0]
F_true_g = true_cdf(xg)
f_true_g = true_pdf(xg)


# ============================================================
# 2. Helpers: KDE / ECDF, CDF, W1, MMD^2
# ============================================================
def silverman_bw(x):
    s = np.std(x, ddof=1)
    n = len(x)
    if n <= 1 or s == 0:
        return 1.0
    return 1.06 * s * n ** (-1 / 5)


def kde_pdf(kde, x1d):
    return np.exp(kde.score_samples(x1d.reshape(-1, 1)))


def cdf_from_pdf_on_grid(pdf_vals, dx, renormalize=True):
    F = np.empty_like(pdf_vals)
    F[0] = 0.0
    F[1:] = np.cumsum((pdf_vals[:-1] + pdf_vals[1:]) * 0.5 * dx)
    if renormalize:
        total = F[-1]
        if total <= 0:
            total = 1.0
        F = F / total
    return F


def w1_from_cdfs_on_grid(F_true, F_model, xg):
    return np.trapezoid(np.abs(F_true - F_model), xg)


def empirical_cdf_on_grid(samples, xg):
    samples_sorted = np.sort(samples)
    n = len(samples_sorted)
    idx = np.searchsorted(samples_sorted, xg, side="right")
    return idx / n




def mmd2_from_pdfs(f_true, f_model, xg, sigma=0.5):
    dx_local = xg[1] - xg[0]
    X = xg[:, None]
    K = np.exp(-(X - X.T) ** 2 / (2.0 * sigma ** 2))
    term11 = np.sum((f_true @ K) * f_true) * dx_local * dx_local
    term22 = np.sum((f_model @ K) * f_model) * dx_local * dx_local
    term12 = np.sum((f_true @ K) * f_model) * dx_local * dx_local
    return term11 + term22 - 2.0 * term12


# ============================================================
# 3. Recursive real + synthetic estimator (KDE or ECDF)
# ============================================================
def run_recursive_estimator(
    p_target=0.5,
    alpha=0.5,          
    m1=50,
    T=200,
    use_ecdf=True,
    h0=0.5,
    bias_rate=0.0,      
):
    rng = np.random.default_rng()  

    if alpha <= 0.0:
        m2 = 0
    else:
        m2 = int(round(alpha * m1 / (1 - alpha)))
    alpha_used = m2 / (m1 + m2) if (m1 + m2) > 0 else 0.0

    X_real = np.empty(0, dtype=float)
    X_synth = np.empty(0, dtype=float)

    Ms = np.empty(T, dtype=int)
    W1s = np.empty(T, dtype=float)
    bandwidths = np.empty(T, dtype=float)
    biases = np.empty(T, dtype=float)

    f_hat_history = []
    F_hat_history = []

    h0_init = None
    M1 = None

    for t in range(1, T + 1):
        idx = t - 1

        bias_t = (0.1) * (t + 5) ** (-bias_rate) if bias_rate > 0 else 0.0
        biases[idx] = bias_t

        x_new_real = sample_true(m1, rng, bias=bias_t)
        X_real = x_new_real if X_real.size == 0 else np.concatenate((X_real, x_new_real))

        X_train = np.concatenate((X_real, X_synth)) if X_synth.size > 0 else X_real
        M_t = X_train.size
        Ms[idx] = M_t

        if use_ecdf:
            F_hat_g = empirical_cdf_on_grid(X_train, xg)
            f_hat_g = np.gradient(F_hat_g, dx)
            bandwidths[idx] = 0.0

            if m2 > 0:
                synth_samples = rng.choice(X_train, size=m2, replace=True)
                X_synth = synth_samples if X_synth.size == 0 else np.concatenate((X_synth, synth_samples))
        else:
            X_train_2d = X_train.reshape(-1, 1)

            if h0_init is None:
                h0_init = h0
                M1 = M_t

            h_t = h0_init * (M_t / M1) ** (-p_target)
            bandwidths[idx] = h_t

            kde = KernelDensity(kernel="gaussian", bandwidth=h_t).fit(X_train_2d)
            f_hat_g = kde_pdf(kde, xg)
            F_hat_g = cdf_from_pdf_on_grid(f_hat_g, dx)

            if m2 > 0:
                synth_samples = kde.sample(m2).ravel()
                X_synth = synth_samples if X_synth.size == 0 else np.concatenate((X_synth, synth_samples))

        W1s[idx] = w1_from_cdfs_on_grid(F_true_g, F_hat_g, xg)
        f_hat_history.append(f_hat_g)
        F_hat_history.append(F_hat_g)

    return {
        "M": Ms,
        "W1": W1s,
        "bandwidths": bandwidths,
        "biases": biases,
        "bias_rate": bias_rate,
        "p_target": p_target,
        "alpha": alpha_used,      
        "m1": m1,
        "m2": m2,
        "use_ecdf": use_ecdf,
        "f_hat_history": np.stack(f_hat_history),
        "F_hat_history": np.stack(F_hat_history),
    }


# ============================================================
# 4. Power-law slope estimation (log–log regression)
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
        return np.nan, np.nan

    logM = np.log(M_use).reshape(-1, 1)
    logV = np.log(vals_use)
    reg = LinearRegression().fit(logM, logV)
    return float(reg.coef_[0]), float(reg.intercept_)


# ============================================================
# 5. JSON serialization helper
# ============================================================
def _make_json_serializable(obj):
    import numpy as _np
    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    if isinstance(obj, (_np.float32, _np.float64)):
        return float(obj)
    if isinstance(obj, (_np.int32, _np.int64)):
        return int(obj)
    return obj


# ============================================================
# 6. Plotting Helpers
# ============================================================
def plot_density_first_rep(f_true_g, f_hat_final, xg, p_val, alpha_real, bias_rate, save_path):
    plt.figure(figsize=(7, 4))
    plt.plot(xg, f_true_g, linewidth=2, label="True density P0")
    plt.plot(xg, f_hat_final, linewidth=2, label="Estimated density")
    plt.xlabel("x")
    plt.ylabel("density")
    plt.title(f"Estimated density (final): p={p_val}, $\\alpha_{{real}}$={alpha_real:.2f}, q={bias_rate:.2f}")
    plt.grid(True, axis="y", ls="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

def plot_with_trend(M, values, metric_name, p_val, alpha_real_val, bias_rate, drop_first=10, save_path=None):
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
    
    # LaTeX formatting for titles and y-axis
    if metric_name == "W1":
        metric_name_tex = r"W_1"
    elif metric_name == "MMD":
        metric_name_tex = r"\mathrm{MMD}"
    else:
        metric_name_tex = metric_name
        
    plt.ylabel(rf"$\log_{{10}}({metric_name_tex})$")
    plt.title(rf"${metric_name_tex}$ Scaling: p={p_val}, $\alpha_{{real}}$={alpha_real_val:.2f}, q={bias_rate:.2f}")
    
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# ============================================================
# 7. Experiment driver
# ============================================================
PLOT_LOCK = threading.Lock()


def combo_label(p_val, alpha_real_val, bias_rate):
    return f"p={p_val:.2f}_alphaReal={alpha_real_val:.2f}_q={bias_rate:.2f}"


def combo_dirname(p_val, alpha_real_val, bias_rate):
    return combo_label(p_val, alpha_real_val, bias_rate).replace("=", "").replace(".", "p")


def save_rep_outputs(
    out,
    MMD2s,
    MMDs,
    W1_for_slope,
    MMDs_for_slope,
    alpha_real,
    alpha_est,
    q,
    p,
    rep,
    drop_first,
    slope_W1,
    intercept_W1,
    slope_MMD,
    intercept_MMD,
    data_dir,
    save_full_history=True,
    save_rep_metrics_csv=True,
):
    """Save one repetition in files that are enough to recompute rates and remake plots."""
    data_dir.mkdir(parents=True, exist_ok=True)

    M = np.asarray(out["M"])
    W1 = np.asarray(out["W1"])
    rep_tag = f"rep={rep:03d}"

    rep_meta = {
        "rep": int(rep),
        "alpha_real": float(alpha_real),
        "alpha_est": float(alpha_est),
        "alpha_est_used": float(out["alpha"]),
        "alpha_real_used": float(1.0 - out["alpha"]),
        "bias_rate": float(q),
        "p_target": float(p),
        "T": int(len(M)),
        "m1": int(out["m1"]),
        "m2": int(out["m2"]),
        "use_ecdf": bool(out["use_ecdf"]),
        "h0": float(h0),
        "drop_first_for_fit": int(drop_first),
        "mmd_sigma": 0.5,
        "slope_W1": float(slope_W1),
        "intercept_W1": float(intercept_W1),
        "slope_MMD": float(slope_MMD),
        "intercept_MMD": float(intercept_MMD),
        "observed_rate_W1": float(-slope_W1),
        "observed_rate_MMD": float(-slope_MMD),
        "theoretical_rate": float(min(p, alpha_real, q)),
        "grid": {"left": float(left), "right": float(right), "m_grid": int(m_grid)},
        "true_mixture": {
            "w1": float(w1),
            "mu1": float(mu1), "sigma1": float(sigma1),
            "mu2": float(mu2), "sigma2": float(sigma2),
            "mu3": float(mu3), "sigma3": float(sigma3),
        },
    }

    npz_path = data_dir / f"{rep_tag}_data.npz"
    npz_payload = {
        "M": M,
        "W1": W1,
        "MMD": np.asarray(MMDs),
        "MMD2": np.asarray(MMD2s),
        "bandwidths": np.asarray(out["bandwidths"]),
        "biases": np.asarray(out["biases"]),
        "W1_for_slope": np.asarray(W1_for_slope),
        "MMD_for_slope": np.asarray(MMDs_for_slope),
        "xg": np.asarray(xg),
        "f_true_g": np.asarray(f_true_g),
        "F_true_g": np.asarray(F_true_g),
        "metadata_json": np.array(json.dumps(rep_meta, sort_keys=True)),
    }
    if save_full_history:
        npz_payload["f_hat_history"] = np.asarray(out["f_hat_history"])
        npz_payload["F_hat_history"] = np.asarray(out["F_hat_history"])
    else:
        npz_payload["f_hat_final"] = np.asarray(out["f_hat_history"][-1])
        npz_payload["F_hat_final"] = np.asarray(out["F_hat_history"][-1])

    np.savez_compressed(npz_path, **npz_payload)

    summary_json_path = data_dir / f"{rep_tag}_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(rep_meta, f, indent=2, sort_keys=True)

    metrics_csv_path = ""
    if save_rep_metrics_csv:
        metrics_csv_path = data_dir / f"{rep_tag}_metrics.csv"
        with open(metrics_csv_path, "w", newline="", encoding="utf-8") as f:
            wcsv = csv.writer(f)
            wcsv.writerow([
                "iter", "M", "W1", "MMD", "MMD2", "bandwidth", "bias",
                "W1_for_slope", "MMD_for_slope",
            ])
            for i in range(len(M)):
                wcsv.writerow([
                    i + 1,
                    int(M[i]),
                    float(W1[i]),
                    float(MMDs[i]) if np.isfinite(MMDs[i]) else "nan",
                    float(MMD2s[i]) if np.isfinite(MMD2s[i]) else "nan",
                    float(out["bandwidths"][i]),
                    float(out["biases"][i]),
                    float(W1_for_slope[i]) if np.isfinite(W1_for_slope[i]) else "nan",
                    float(MMDs_for_slope[i]) if np.isfinite(MMDs_for_slope[i]) else "nan",
                ])

    return {
        **rep_meta,
        "npz_path": str(npz_path),
        "summary_json_path": str(summary_json_path),
        "metrics_csv_path": str(metrics_csv_path),
    }


def run_one(
    alpha_real,
    q,
    p,
    rep,
    T,
    m1,
    h0,
    drop_first,
    data_root,
    density_root,
    rate_root,
    save_full_history,
    save_rep_metrics_csv,
    rep_density_mode,
    rep_rate_mode,
):
    alpha_est = 1 - alpha_real
    out = run_recursive_estimator(
        p_target=p,
        alpha=alpha_est,
        m1=m1,
        T=T,
        use_ecdf=True,
        h0=h0,
        bias_rate=q
    )

    M = out["M"]
    W1 = out["W1"]
    f_hist = out["f_hat_history"]

    MMD2s = np.array([mmd2_from_pdfs(f_true_g, f_hist[t], xg, 0.5) for t in range(T)], dtype=float)

    # Compute MMD safely.
    MMDs = np.full_like(MMD2s, np.nan)
    valid_mmd = MMD2s > 0
    MMDs[valid_mmd] = np.sqrt(MMD2s[valid_mmd])

    # Log normalization correction, unchanged from the original script.
    if np.isclose(p, alpha_real, atol=1e-8):
        logM = np.log(np.maximum(M, 2))
        W1_for_slope = W1 / logM
        MMDs_for_slope = MMDs / logM
    else:
        W1_for_slope = W1
        MMDs_for_slope = MMDs

    sW1, bW1 = estimate_power_law_slope(M, W1_for_slope, drop_first=drop_first)
    sMMD, bMMD = estimate_power_law_slope(M, MMDs_for_slope, drop_first=drop_first)

    dirname = combo_dirname(p, alpha_real, q)
    label = combo_label(p, alpha_real, q)
    data_dir = Path(data_root) / dirname
    dens_dir = Path(density_root) / dirname
    rate_dir = Path(rate_root) / dirname
    data_dir.mkdir(parents=True, exist_ok=True)
    dens_dir.mkdir(parents=True, exist_ok=True)
    rate_dir.mkdir(parents=True, exist_ok=True)

    rep_record = save_rep_outputs(
        out=out,
        MMD2s=MMD2s,
        MMDs=MMDs,
        W1_for_slope=W1_for_slope,
        MMDs_for_slope=MMDs_for_slope,
        alpha_real=alpha_real,
        alpha_est=alpha_est,
        q=q,
        p=p,
        rep=rep,
        drop_first=drop_first,
        slope_W1=sW1,
        intercept_W1=bW1,
        slope_MMD=sMMD,
        intercept_MMD=bMMD,
        data_dir=data_dir,
        save_full_history=save_full_history,
        save_rep_metrics_csv=save_rep_metrics_csv,
    )

    do_save_dens = (rep_density_mode == "all") or (rep_density_mode == "first" and rep == 0)
    if do_save_dens:
        dens_path = dens_dir / f"density_rep={rep:03d}_{label}.png"
        with PLOT_LOCK:
            plot_density_first_rep(f_true_g, f_hist[-1], xg, p, alpha_real, q, dens_path)
    else:
        dens_path = ""

    do_save_rate = (rep_rate_mode == "all") or (rep_rate_mode == "first" and rep == 0)
    if do_save_rate:
        w1_rate_path = rate_dir / f"W1_rep={rep:03d}_{label}.png"
        mmd_rate_path = rate_dir / f"MMD_rep={rep:03d}_{label}.png"
        with PLOT_LOCK:
            plot_with_trend(M, W1_for_slope, "W1", p, alpha_real, q, drop_first, w1_rate_path)
            plot_with_trend(M, MMDs_for_slope, "MMD", p, alpha_real, q, drop_first, mmd_rate_path)
    else:
        w1_rate_path = ""
        mmd_rate_path = ""

    rep_record.update({
        "combo_label": label,
        "density_plot_path": str(dens_path),
        "W1_rate_plot_path": str(w1_rate_path),
        "MMD_rate_plot_path": str(mmd_rate_path),
    })

    return rep_record


if __name__ == "__main__":
    p_values = [0.5]
    alphas = [0.25, 0.5, 0.75]
    bias_rates = [0.25, 0.5, 0.75]
    T, m1, n_reps, h0, drop_first = 3000, 50, 100, 0.5, 200

    # Saving controls only; these do not change the experiment itself.
    # SAVE_FULL_HISTORY=True stores f_hat_history and F_hat_history for every rep,
    # enough to remake density curves at every iteration, but the files can be large.
    SAVE_FULL_HISTORY = True
    SAVE_REP_METRICS_CSV = True
    REP_DENS_MODE = "all"   # "first" preserves the original plotting volume; set to "all" for every rep.
    REP_RATE_MODE = "all"   # "first" preserves the original plotting volume; set to "all" for every rep.

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"biased_alphaReal_h0={h0}_T={T}_m1={m1}_nreps={n_reps}_{ts}"
    out_dir = Path("logs/ECDF") / run_name
    fig_dir = out_dir / "figures"
    density_root = fig_dir / "density_plots"
    rate_root = fig_dir / "rate_plots"
    data_root = out_dir / "rep_data"

    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    density_root.mkdir(parents=True, exist_ok=True)
    rate_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)

    # Centralized logging configuration
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

    def log_config(title: str, cfg: dict):
        log("\n" + "-" * 80)
        log(title)
        log(json.dumps(cfg, indent=2, sort_keys=True))
        log("-" * 80 + "\n")

    RUN_CONFIG = {
        "p_values": p_values,
        "alphas_real": alphas,
        "bias_rates": bias_rates,
        "T": T,
        "m1": m1,
        "n_reps": n_reps,
        "h0": h0,
        "drop_first_for_fit": drop_first,
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
            "mu3": float(mu3), "sigma3": float(sigma3)
        }
    }
    log_config("RUN CONFIG", RUN_CONFIG)

    tasks = [
        (a, q, p, r, T, m1, h0, drop_first,
         data_root, density_root, rate_root,
         SAVE_FULL_HISTORY, SAVE_REP_METRICS_CSV, REP_DENS_MODE, REP_RATE_MODE)
        for a in alphas for q in bias_rates for p in p_values for r in range(n_reps)
    ]
    log(f"Total tasks={len(tasks)}")

    rep_results = []
    # Using ThreadPoolExecutor for speed, but each rep writes its own files immediately.
    with ThreadPoolExecutor(max_workers=18) as ex:
        futs = [ex.submit(run_one, *t) for t in tasks]
        for i, f in enumerate(as_completed(futs)):
            rep_record = f.result()
            rep_results.append(rep_record)
            log(
                f"Completed {i + 1}/{len(tasks)} | "
                f"{rep_record['combo_label']} rep={rep_record['rep']:03d} "
                f"slopes: W1={rep_record['slope_W1']:.4f}, "
                f"MMD={rep_record['slope_MMD']:.4f} | "
                f"data={rep_record['npz_path']}"
            )

    all_results = []
    summary = []
    keys = sorted({(r["alpha_real"], r["bias_rate"], r["p_target"]) for r in rep_results})

    for alpha_real, q, p in keys:
        grp = [r for r in rep_results if r["alpha_real"] == alpha_real and r["bias_rate"] == q and r["p_target"] == p]
        grp = sorted(grp, key=lambda r: r["rep"])

        slopes_W1 = np.array([r["slope_W1"] for r in grp], dtype=float)
        slopes_MMD = np.array([r["slope_MMD"] for r in grp], dtype=float)

        log("\n" + "=" * 70)
        log(f"Summarizing {len(grp)}×: p={p}, alpha_real={alpha_real:.2f}, q={q:.2f}")

        mean_W1 = float(np.nanmean(slopes_W1))
        std_W1  = float(np.nanstd(slopes_W1, ddof=1)) if len(slopes_W1) > 1 else 0.0
        mean_MMD = float(np.nanmean(slopes_MMD))
        std_MMD  = float(np.nanstd(slopes_MMD, ddof=1)) if len(slopes_MMD) > 1 else 0.0

        median_W1 = float(np.nanmedian(slopes_W1))
        median_MMD = float(np.nanmedian(slopes_MMD))

        theo = float(min(p, alpha_real, q))

        log(f"*** avg over {len(grp)} ***")
        log(f"mean slope W1={mean_W1:.3f} (std={std_W1:.3f}) | observed rate≈{-mean_W1:.3f}")
        log(f"mean slope MMD={mean_MMD:.3f} (std={std_MMD:.3f}) | observed rate≈{-mean_MMD:.3f}")
        log(f"theory rate min(p,alpha_real,q)={theo:.3f}")

        summary.append({
            "alpha_real": alpha_real,
            "bias_rate": q,
            "p_target": p,
            "theoretical_rate": theo,
            "observed_rate_mean_W1": -mean_W1,
            "observed_rate_median_W1": -median_W1,
            "observed_rate_std_W1": float(np.nanstd(-slopes_W1, ddof=1)) if len(slopes_W1) > 1 else 0.0,
            "observed_rate_mean_MMD": -mean_MMD,
            "observed_rate_median_MMD": -median_MMD,
            "observed_rate_std_MMD": float(np.nanstd(-slopes_MMD, ddof=1)) if len(slopes_MMD) > 1 else 0.0,
        })

        all_results.append(dict(
            combo_label=combo_label(p, alpha_real, q),
            alpha_real=float(alpha_real),
            bias_rate=float(q),
            p_target=float(p),
            T=int(T),
            m1=int(m1),
            m2=int(grp[0]["m2"]),
            drop_first_for_fit=int(drop_first),
            slopes_W1=slopes_W1.tolist(),
            slopes_MMD=slopes_MMD.tolist(),
            mean_slope_W1=float(mean_W1),
            mean_slope_MMD=float(mean_MMD),
            std_slope_W1=float(std_W1),
            std_slope_MMD=float(std_MMD),
            median_slope_W1=float(median_W1),
            median_slope_MMD=float(median_MMD),
            theoretical_rate=float(theo),
            observed_rate_mean_W1=float(-mean_W1),
            observed_rate_median_W1=float(-median_W1),
            observed_rate_mean_MMD=float(-mean_MMD),
            observed_rate_median_MMD=float(-median_MMD),
            rep_data=grp,
        ))

    # Build Summary Table (now including standard deviations and all metrics)
    summary_path = out_dir / "summary_table.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        wcsv = csv.writer(f)
        wcsv.writerow([
            "alpha_real", "bias_rate", "p_target", "T", "m1", "n_reps", "h0", "drop_first_for_fit",
            "theoretical_rate",
            "observed_rate_mean_W1", "observed_rate_median_W1", "observed_rate_std_W1",
            "observed_rate_mean_MMD", "observed_rate_median_MMD", "observed_rate_std_MMD",
        ])
        for s in summary:
            wcsv.writerow([
                s["alpha_real"], s["bias_rate"], s["p_target"], T, m1, n_reps, h0, drop_first,
                s["theoretical_rate"],
                s["observed_rate_mean_W1"], s["observed_rate_median_W1"], s["observed_rate_std_W1"],
                s["observed_rate_mean_MMD"], s["observed_rate_median_MMD"], s["observed_rate_std_MMD"],
            ])
    log(f"Saved summary_table.csv -> {summary_path}")

    rep_index_path = out_dir / "all_rep_index.csv"
    with open(rep_index_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "combo_label", "rep", "alpha_real", "alpha_est", "alpha_est_used", "alpha_real_used",
            "bias_rate", "p_target", "T", "m1", "m2", "use_ecdf", "h0", "drop_first_for_fit",
            "mmd_sigma", "slope_W1", "intercept_W1",
            "slope_MMD", "intercept_MMD", "observed_rate_W1",
            "observed_rate_MMD", "theoretical_rate", "npz_path", "summary_json_path",
            "metrics_csv_path", "density_plot_path", "W1_rate_plot_path",
            "MMD_rate_plot_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(rep_results, key=lambda r: (r["alpha_real"], r["bias_rate"], r["p_target"], r["rep"])):
            writer.writerow(row)
    log(f"Saved per-repetition index -> {rep_index_path}")

    with open(out_dir / "all_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "RUN_CONFIG": RUN_CONFIG,
            "results": [{k: _make_json_serializable(v) for k, v in r.items()} for r in all_results],
        }, f, indent=2)
    log(f"Saved combination-level JSON to {out_dir / 'all_results.json'}")

    with open(out_dir / "all_rep_index.json", "w", encoding="utf-8") as f:
        json.dump(rep_results, f, indent=2)
    log(f"Saved per-repetition JSON index to {out_dir / 'all_rep_index.json'}")

    log("\nDone. Saved enough per-repetition data to recompute slopes/rates and remake plots.")
