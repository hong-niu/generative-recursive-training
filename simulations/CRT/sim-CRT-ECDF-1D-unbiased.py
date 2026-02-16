"""
Recursive density estimation under real–synthetic data mixing.

We study a recursive kernel density estimator with 0 bandwidth,
corresponding to the empirical CDF (ECDF) on a fixed grid, where
at each iteration n:

  1. A fixed batch of m1 new real samples is drawn from a stationary
     two-component Gaussian mixture.
  2. Synthetic samples are generated from the current estimator.
  3. The estimator is recomputed from all accumulated data.

The mixing proportion is controlled by alpha ∈ [0, 1], interpreted as the
target fraction of real samples in the training distribution. In the
implementation, alpha determines the number of synthetic samples m2 added
per iteration so that alpha ≈ m1 / (m1 + m2), up to integer rounding.

We evaluate convergence to the ground-truth distribution using:
    • Wasserstein-1 distance (W1)
    • Maximum mean discrepancy (MMD²), computed on a fixed grid

Power-law scaling exponents are estimated via log–log regression, and
empirical slopes are compared to the theoretical exponent min(p, alpha),
up to logarithmic corrections.

"""

import os
import sys
import json
import csv
import datetime
import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import ndtr
from sklearn.linear_model import LinearRegression

import torch

# ============================================================
# 1. Data-generating distribution: stationary two-component Gaussian mixture
# ============================================================
w1 = 0.35
mu1, sigma1 = -2.0, 0.8
mu2, sigma2 = 1.0, 1.3


def sample_true_np(n, rng):
    """Draw n i.i.d. samples from the ground-truth mixture (NumPy)."""
    if n <= 0:
        return np.empty(0, dtype=float)
    z = rng.random(n)
    n1 = np.count_nonzero(z < w1)
    n2 = n - n1
    return np.concatenate([
        rng.normal(mu1, sigma1, size=n1),
        rng.normal(mu2, sigma2, size=n2),
    ])


def true_pdf_np(x):
    """Ground-truth mixture density evaluated at x (NumPy)."""
    return w1 * norm.pdf(x, mu1, sigma1) + (1 - w1) * norm.pdf(x, mu2, sigma2)


def true_cdf_np(x):
    """Ground-truth mixture CDF evaluated at x (NumPy)."""
    return w1 * ndtr((x - mu1) / sigma1) + (1 - w1) * ndtr((x - mu2) / sigma2)


# Torch implementations (GPU-compatible)
def true_pdf_torch(x_t):
    """Ground-truth mixture density evaluated at x_t (torch tensor)."""
    coef1 = w1 / (sigma1 * np.sqrt(2.0 * np.pi))
    coef2 = (1.0 - w1) / (sigma2 * np.sqrt(2.0 * np.pi))
    z1 = (x_t - mu1) / sigma1
    z2 = (x_t - mu2) / sigma2
    return coef1 * torch.exp(-0.5 * z1**2) + coef2 * torch.exp(-0.5 * z2**2)


def true_cdf_torch(x_t):
    """Ground-truth mixture CDF evaluated at x_t via erf (torch tensor)."""
    z1 = (x_t - mu1) / sigma1
    z2 = (x_t - mu2) / sigma2
    Phi1 = 0.5 * (1.0 + torch.erf(z1 / np.sqrt(2.0)))
    Phi2 = 0.5 * (1.0 + torch.erf(z2 / np.sqrt(2.0)))
    return w1 * Phi1 + (1.0 - w1) * Phi2


def sample_true_torch(n, device, dtype=torch.float64):
    """Draw n i.i.d. samples from the mixture directly on the given torch device."""
    if n <= 0:
        return torch.empty(0, device=device, dtype=dtype)
    z = torch.rand(n, device=device, dtype=dtype)
    n1 = (z < w1).sum().item()
    n2 = n - n1
    x1 = torch.randn(n1, device=device, dtype=dtype) * sigma1 + mu1
    x2 = torch.randn(n2, device=device, dtype=dtype) * sigma2 + mu2
    return torch.cat([x1, x2], dim=0)


# Fixed evaluation grid (used for deterministic W1/MMD computations)
left = min(mu1 - 6 * sigma1, mu2 - 6 * sigma2)
right = max(mu1 + 6 * sigma1, mu2 + 6 * sigma2)
m_grid = 500

xg = np.linspace(left, right, m_grid)
dx = xg[1] - xg[0]
F_true_g = true_cdf_np(xg)
f_true_g = true_pdf_np(xg)

# Torch configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64

xg_t = torch.from_numpy(xg).to(device=DEVICE, dtype=DTYPE)
dx_t = xg_t[1] - xg_t[0]
F_true_g_t = torch.from_numpy(F_true_g).to(device=DEVICE, dtype=DTYPE)
f_true_g_t = torch.from_numpy(f_true_g).to(device=DEVICE, dtype=DTYPE)

# Precompute Gaussian kernel matrix for MMD^2 on the grid
MMD_SIGMA = 0.5
Xgrid = xg_t.view(-1, 1)
K_mmd = torch.exp(-(Xgrid - Xgrid.T) ** 2 / (2.0 * MMD_SIGMA**2))


# ============================================================
# 2. Numerical helpers and distributional metrics (torch)
# ============================================================
def cdf_from_pdf_on_grid_torch(pdf_vals, dx, renormalize=True):
    """
    Compute a CDF from density values on a uniform grid via trapezoidal integration.

    Returns a torch tensor F with F[0] = 0. If renormalize is True, scales
    the result so that F[-1] = 1 (up to numerical precision).
    """
    F = torch.empty_like(pdf_vals)
    F[0] = 0.0
    F[1:] = torch.cumsum((pdf_vals[:-1] + pdf_vals[1:]) * 0.5 * dx, dim=0)
    if renormalize:
        total = F[-1].clamp_min(1e-12)
        F = F / total
    return F


def w1_from_cdfs_on_grid_torch(F_true, F_model, xg_t):
    """Compute Wasserstein-1 distance via ∫ |F_true(x) - F_model(x)| dx."""
    diff = torch.abs(F_true - F_model)
    return torch.trapz(diff, xg_t)


def empirical_cdf_on_grid_torch(samples, xg_t, dtype):
    """Empirical CDF evaluated on xg_t: F_hat(x) = (1/n) Σ 1{X_i ≤ x}."""
    sorted_samples, _ = torch.sort(samples)
    n = sorted_samples.shape[0]
    idx = torch.searchsorted(sorted_samples, xg_t, right=True)
    return idx.to(dtype=dtype) / float(n)


def ecdf_pdf_from_cdf_fd_torch(F_hat_g_t, dx_t):
    """
    Stable finite-difference derivative of ECDF CDF values on a uniform grid.
    Central differences in the interior, one-sided at boundaries.
    """
    f_hat_g_t = torch.empty_like(F_hat_g_t)
    # interior
    f_hat_g_t[1:-1] = (F_hat_g_t[2:] - F_hat_g_t[:-2]) / (2.0 * dx_t)
    # boundaries
    f_hat_g_t[0] = (F_hat_g_t[1] - F_hat_g_t[0]) / dx_t
    f_hat_g_t[-1] = (F_hat_g_t[-1] - F_hat_g_t[-2]) / dx_t
    # safety
    f_hat_g_t = torch.nan_to_num(f_hat_g_t, nan=0.0, posinf=0.0, neginf=0.0)
    return f_hat_g_t


def mmd2_from_pdfs_torch(f_true, f_model, K, dx):
    """
    Compute MMD^2 between two densities on a fixed grid.

    The kernel matrix K is assumed to be precomputed on the grid.
    """
    v_true = torch.mv(K, f_true)
    v_model = torch.mv(K, f_model)
    dx2 = dx * dx
    term11 = torch.dot(f_true, v_true) * dx2
    term22 = torch.dot(f_model, v_model) * dx2
    term12 = torch.dot(f_true, v_model) * dx2
    return term11 + term22 - 2.0 * term12


# ============================================================
# 3. Recursive density estimator with real–synthetic mixing
# ============================================================
def run_recursive_estimator(
    p_target=0.5,
    alpha=0.5,      # target real-data fraction
    m1=50,
    T=200,
    use_ecdf=False,
    h0=0.5,
):
    """
    Run the recursive density estimation procedure (GPU-accelerated).

    At each iteration n:
      (i)   add m1 new real samples to the real pool,
      (ii)  form the training set as all accumulated real + synthetic samples,
      (iii) estimate density/CDF on a fixed grid,
      (iv)  generate m2 additional synthetic samples (if m2 > 0).

    The mixing parameter alpha is interpreted as the target real-data fraction.
    The synthetic count m2 is chosen so that alpha ≈ m1 / (m1 + m2), up to
    integer rounding.

    Returns metrics: W1 and MMD^2 only (TV removed).
    """
    device = DEVICE
    dtype = DTYPE

    # Choose m2 (synthetic samples per iteration) from target alpha
    if alpha >= 1.0:
        m2 = 0
        alpha = 1.0
    elif alpha <= 0.0:
        m2 = 0
        alpha = 0.0
    else:
        m2 = int(round((1.0 - alpha) * m1 / alpha))

    # Realized fraction after rounding m2 to an integer
    alpha_eff = (m1 / (m1 + m2)) if (m1 + m2) > 0 else 0.0

    X_real = torch.empty(0, device=device, dtype=dtype)
    X_synth = torch.empty(0, device=device, dtype=dtype)

    Ms = np.empty(T, dtype=int)
    W1s = np.empty(T, dtype=float)
    MMD2s = np.empty(T, dtype=float)
    bandwidths = np.empty(T, dtype=float)

    f_hat_history = []
    F_hat_history = []

    normal_const = 1.0 / np.sqrt(2.0 * np.pi)

    with torch.no_grad():
        for n in range(1, T + 1):
            idx = n - 1

            # (1) Add new real samples
            x_new_real = sample_true_torch(m1, device=device, dtype=dtype)
            X_real = torch.cat((X_real, x_new_real), dim=0)

            # (2) Form training set
            X_train = torch.cat((X_real, X_synth), dim=0) if X_synth.numel() > 0 else X_real
            M_n = X_train.shape[0]
            Ms[idx] = int(M_n)

            # (3) Estimate density/CDF + generate synthetic samples
            if use_ecdf:
                F_hat_g_t = empirical_cdf_on_grid_torch(X_train, xg_t, dtype=dtype)
                f_hat_g_t = ecdf_pdf_from_cdf_fd_torch(F_hat_g_t, dx_t)
                bandwidths[idx] = 0.0

                if m2 > 0:
                    idx_resample = torch.randint(0, M_n, (m2,), device=device)
                    synth_samples = X_train[idx_resample]
                    X_synth = torch.cat((X_synth, synth_samples), dim=0)

            else:
                h_n = h0 * (n) ** (-p_target / 2.0)
                bandwidths[idx] = float(h_n)

                h_n_t = torch.tensor(h_n, device=device, dtype=dtype)
                diff = (xg_t.unsqueeze(1) - X_train.unsqueeze(0)) / h_n_t
                kernel_vals = torch.exp(-0.5 * diff**2)
                f_hat_g_t = kernel_vals.mean(dim=1) * (normal_const / h_n_t)
                F_hat_g_t = cdf_from_pdf_on_grid_torch(f_hat_g_t, dx_t, renormalize=True)

                if m2 > 0:
                    idx_resample = torch.randint(0, M_n, (m2,), device=device)
                    base_samples = X_train[idx_resample]
                    noise = torch.randn(m2, device=device, dtype=dtype) * h_n_t
                    synth_samples = base_samples + noise
                    X_synth = torch.cat((X_synth, synth_samples), dim=0)

            # (4) Metrics on the fixed grid (W1 + MMD^2 only)
            W1_n_t = w1_from_cdfs_on_grid_torch(F_true_g_t, F_hat_g_t, xg_t)
            MMD2_n_t = mmd2_from_pdfs_torch(f_true_g_t, f_hat_g_t, K_mmd, float(dx))

            W1s[idx] = float(W1_n_t.cpu().item())
            MMD2s[idx] = float(MMD2_n_t.cpu().item())

            f_hat_history.append(f_hat_g_t.cpu().numpy())
            F_hat_history.append(F_hat_g_t.cpu().numpy())

            if (n == 1) or (n % 50 == 0):
                if use_ecdf:
                    print(
                        f"[p={p_target:.2f} (ECDF), alpha={alpha:.2f}] "
                        f"n={n:4d} | M_n={M_n:6d} | W1={W1s[idx]:.6f}"
                    )
                else:
                    print(
                        f"[p={p_target:.2f}, alpha={alpha:.2f}] "
                        f"n={n:4d} | M_n={M_n:6d} | h_n={h_n:.4f} | W1={W1s[idx]:.6f}"
                    )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return {
        "M": Ms,
        "W1": W1s,
        "MMD2": MMD2s,
        "bandwidths": bandwidths,
        "p_target": p_target,
        "alpha": float(alpha),          # requested target real fraction
        "alpha_eff": float(alpha_eff),  # realized fraction after rounding
        "m1": m1,
        "m2": m2,
        "use_ecdf": use_ecdf,
        "f_hat_history": np.stack(f_hat_history),
        "F_hat_history": np.stack(F_hat_history),
    }


# ============================================================
# 4. Power-law slope estimator
# ============================================================
def estimate_power_law_slope(M, vals, drop_first=10, eps=1e-30):
    """
    Estimate a power-law exponent by fitting log(vals) = a + b log(M).

    Returns
    -------
    slope b and intercept a, corresponding to vals ~ M^b.

    Robustifies against non-finite and non-positive values by filtering.
    """
    M = np.asarray(M, dtype=float)
    vals = np.asarray(vals, dtype=float)

    if drop_first >= len(M) - 5:
        drop_first = max(0, len(M) - 5)

    M_use = M[drop_first:]
    vals_use = vals[drop_first:] + eps

    mask = np.isfinite(M_use) & np.isfinite(vals_use) & (M_use > 0) & (vals_use > 0)
    M_use = M_use[mask]
    vals_use = vals_use[mask]

    if len(M_use) < 5:
        return np.nan, np.nan

    logM = np.log(M_use).reshape(-1, 1)
    logV = np.log(vals_use)
    reg = LinearRegression().fit(logM, logV)
    return float(reg.coef_[0]), float(reg.intercept_)


def _make_json_serializable(obj):
    """Convert NumPy arrays/scalars into JSON-serializable Python types."""
    import numpy as _np
    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    if isinstance(obj, (_np.float32, _np.float64)):
        return float(obj)
    if isinstance(obj, (_np.int32, _np.int64)):
        return int(obj)
    return obj


# ============================================================
# 5. Experiment driver (sweeps, plotting, and serialization)
# ============================================================
if __name__ == "__main__":
    p_values = [0.5]

    # alpha denotes the fraction of real data in the mixed training distribution
    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    T = 2000
    m1 = 50
    n_reps = 100
    h0 = 0.5

    USE_ERROR_BARS = False

    # Output directories and logging
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")
    run_name = f"alpha_h0={h0}_T={T}_m1={m1}_nreps={n_reps}_{timestamp}"
    out_dir = Path("logs/ECDF/") / run_name
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    log_file = out_dir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    def log(msg: str):
        logging.info(msg)

    all_results = []

    def plot_density_first_rep(f_true_g, f_hat_final, xg, p_val, alpha_val, save_path):
        """Plot ground-truth density and final estimated density (first repetition only)."""
        plt.figure(figsize=(7, 4))
        plt.plot(xg, f_true_g, linewidth=2, label="True density")
        plt.plot(xg, f_hat_final, linewidth=2, label="Estimated density")
        plt.xlabel("x")
        plt.ylabel("density")
        plt.title(rf"Estimated density: p={p_val}, $\alpha$={alpha_val:.2f}")
        plt.grid(True, axis="y", ls="--", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def plot_with_trend(M, values, metric_name, p_val, alpha_val,
                        drop_first=10, save_path=None):
        """
        Plot log10(values) vs log10(M), with an optional linear fit overlay.

        The fit is performed on the tail segment starting at index drop_first.
        """
        eps = 1e-20
        log_M = np.log10(M)
        log_val = np.log10(values + eps)

        plt.figure(figsize=(6, 4))
        plt.plot(log_M, log_val, "o-", alpha=0.5, markersize=3, label="Distr. Loss")

        if len(M) > drop_first + 5:
            x_fit = log_M[drop_first:]
            y_fit = log_val[drop_first:]
            slope, intercept = np.polyfit(x_fit, y_fit, 1)
            y_pred = slope * x_fit + intercept
            plt.plot(x_fit, y_pred, "r--", linewidth=2, label=f"Fit slope: {slope:.3f}")

        plt.xlabel(r"$\log_{10}$(M$_n$)")
        if metric_name == "W1":
            metric_name_tex = r"W_1"
            plt.title(rf"${metric_name_tex}$ Scaling: p={p_val}, $\alpha$={alpha_val:.2f}")
            plt.ylabel(rf"$\log_{{10}}({metric_name_tex})$")
        else:
            plt.title(rf"{metric_name} Scaling: p={p_val}, $\alpha$={alpha_val:.2f}")
            plt.ylabel(rf"$\log_{{10}}$({metric_name})")

        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.3)
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    # Main sweeps
    for alpha in alphas:
        log("\n" + "#" * 80)
        log(f"=== Running all p for alpha={alpha:.2f} (target real fraction) ===")

        for p in p_values:
            log("\n" + "=" * 70)
            log(f"Running recursive estimator {n_reps}×: p_target={p}, alpha={alpha:.2f}")

            slopes_W1 = []
            slopes_MMD2 = []

            last_out = None
            drop_first = 30

            for rep in range(n_reps):
                log(f"\n--- Repetition {rep + 1}/{n_reps} ---")
                out = run_recursive_estimator(
                    p_target=p,
                    alpha=alpha,
                    m1=m1,
                    T=T,
                    use_ecdf=True,
                    h0=h0,
                )

                M_vals = out["M"]
                W1_vals = out["W1"]
                MMD2s = out["MMD2"]

                # Density plot from the first repetition only (per alpha)
                if rep == 0:
                    if "f_hat_history" in out:
                        f_hat_final = out["f_hat_history"][-1]
                        dens_path = fig_dir / f"density_firstRep_p={p:.2f}_alpha={alpha:.2f}.png"
                        plot_density_first_rep(f_true_g, f_hat_final, xg, p, alpha, dens_path)
                        log(f"Saved density plot -> {dens_path}")
                    else:
                        log("NOTE: density plot skipped (out does not contain 'f_hat_history').")

                # Logarithmic correction in the critical regime p = alpha
                if np.isclose(p, alpha, atol=1e-8):
                    logM = np.log(np.maximum(M_vals, 2.0))
                    W1_for_slope = W1_vals / logM
                    MMD2_for_slope = MMD2s / (logM ** 2)
                else:
                    W1_for_slope = W1_vals
                    MMD2_for_slope = MMD2s

                slope_W1, _ = estimate_power_law_slope(M_vals, W1_for_slope, drop_first=drop_first)
                slope_MMD2, _ = estimate_power_law_slope(M_vals, MMD2_for_slope, drop_first=drop_first)

                log(
                    f"Rep {rep + 1}: slope_W1={slope_W1:.3f}, slope_MMD2={slope_MMD2:.3f}"
                )

                slopes_W1.append(slope_W1)
                slopes_MMD2.append(slope_MMD2)

                last_out = out
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            slopes_W1 = np.array(slopes_W1)
            slopes_MMD2 = np.array(slopes_MMD2)

            mean_W1, std_W1 = slopes_W1.mean(), slopes_W1.std(ddof=1)
            mean_MMD2, std_MMD2 = slopes_MMD2.mean(), slopes_MMD2.std(ddof=1)

            mean_MMD = mean_MMD2 / 2.0
            std_MMD = std_MMD2 / 2.0

            theo_exp = min(p, alpha)

            log(f"\n*** Averaged slopes over {n_reps} runs ***")
            log(f"Mean slope W1    ~ M^{mean_W1:.3f}  (std={std_W1:.3f})")
            log(f"Mean slope MMD   ~ M^{mean_MMD:.3f} (std={std_MMD:.3f})")
            log(f"Mean slope MMD^2 ~ M^{mean_MMD2:.3f} (std={std_MMD2:.3f})")
            log(f"Theoretical exponent (up to logs): min(p, alpha) = {theo_exp:.3f}")
            if np.isclose(p, alpha, atol=1e-8):
                log("Critical regime: metrics were normalized by log(M_n) or log(M_n)^2 prior to slope fitting.")

            # Scaling plots from the last repetition only
            M_vals = last_out["M"]
            W1_vals = last_out["W1"]
            MMD2s = last_out["MMD2"]

            run_label = f"p={p:.2f}_alpha={alpha:.2f}"

            plot_with_trend(
                M_vals, W1_vals, "W1", p, alpha,
                drop_first=drop_first,
                save_path=fig_dir / f"{run_label}_W1.png",
            )

            plot_with_trend(
                M_vals, MMD2s, "MMD^2", p, alpha,
                drop_first=drop_first,
                save_path=fig_dir / f"{run_label}_MMD2.png",
            )

            all_results.append({
                **last_out,
                "p_target": p,
                "alpha": float(alpha),
                "alpha_eff": float(last_out["alpha_eff"]),
                "slopes_W1": slopes_W1,
                "slopes_MMD2": slopes_MMD2,
                "mean_slope_W1": mean_W1,
                "mean_slope_MMD": mean_MMD,
                "mean_slope_MMD2": mean_MMD2,
            })

    # Summary plots: alpha vs mean slope
    unique_ps = sorted(set(res["p_target"] for res in all_results))

    def plot_alpha_vs_mean_slope(
        metric_key,
        slopes_key,
        metric_label,
        filename=None,
        std_factor=1.0,
        use_error_bars=USE_ERROR_BARS,
        ci_mult=1.96,
    ):
        """Plot mean estimated slope versus alpha, alongside the min(p, alpha) reference curve."""
        plt.figure(figsize=(6, 4))
        DARK_BLUE = "#244B99"
        RED = "#CC0000"

        if metric_label == "W1":
            metric_label_tex = r"W_1"
        else:
            metric_label_tex = metric_label

        for p in unique_ps:
            subset = [r for r in all_results if r["p_target"] == p]
            alpha_plot = np.array([r["alpha"] for r in subset], dtype=float)
            means = np.array([r[metric_key] for r in subset], dtype=float)

            order = np.argsort(alpha_plot)
            alpha_plot = alpha_plot[order]
            means = means[order]

            # Plot convention: slopes are displayed as positive magnitudes
            vals_plot = -means

            if use_error_bars:
                stds = np.array([
                    np.array(r[slopes_key]).std(ddof=1) * std_factor
                    for r in subset
                ], dtype=float)
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
        if metric_label_tex == "W_1":
            plt.ylabel(rf"Mean Slope ${metric_label_tex}$")
            plt.title(rf"Mean ${metric_label_tex}$ Slope vs $\alpha$")
        else:
            plt.ylabel(rf"Mean Slope {metric_label_tex}")
            plt.title(rf"Mean {metric_label_tex} Slope vs $\alpha$")

        plt.grid(True, axis="y", ls="--", alpha=0.3)

        # Fix y-axis limits for comparability across plots
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

    plot_alpha_vs_mean_slope(
        metric_key="mean_slope_W1",
        slopes_key="slopes_W1",
        metric_label="W1",
        filename="alpha_vs_mean_slope_W1.png",
    )

    # Note: MMD exponent equals half of the MMD^2 exponent; std_factor accounts for this scaling
    plot_alpha_vs_mean_slope(
        metric_key="mean_slope_MMD",
        slopes_key="slopes_MMD2",
        metric_label="MMD",
        filename="alpha_vs_mean_slope_MMD.png",
        std_factor=0.5,
    )

    plot_alpha_vs_mean_slope(
        metric_key="mean_slope_MMD2",
        slopes_key="slopes_MMD2",
        metric_label="MMD2",
        filename="alpha_vs_mean_slope_MMD2.png",
    )

    # Save JSON
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
        writer.writerow([
            "p_target",
            "alpha (real fraction, requested)",
            "alpha_eff (real fraction, due to rounding)",
            "m1",
            "m2",
            "mean_slope_W1",
            "mean_slope_MMD",
            "mean_slope_MMD2",
        ])
        for res in all_results:
            writer.writerow([
                res["p_target"],
                res["alpha"],
                res.get("alpha_eff", np.nan),
                res["m1"],
                res["m2"],
                res["mean_slope_W1"],
                res["mean_slope_MMD"],
                res["mean_slope_MMD2"],
            ])
    log(f"Saved summary CSV to {summary_path}")
