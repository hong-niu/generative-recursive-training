"""
Recursive density estimation under real–synthetic mixing (biased-real variant).

This script runs a recursive estimator over T iterations:
  1) Draw m1 new real samples from a time-varying biased distribution P_t.
  2) Draw m2 synthetic samples from the current estimator.
  3) Refit the estimator on the accumulated dataset (all past real + synthetic).
  4) Evaluate convergence to the fixed target distribution P0 (unbiased mixture).

Key knobs:
  - p_target: bandwidth decay exponent (KDE branch)
  - alpha: synthetic fraction used to set m2 from m1
  - bias_rate (q): bias schedule exponent via bias_t = 0.1 * (t+5)^(-q)

Metrics:
  - W1 (via CDF difference on a fixed grid)
  - TV (via L1 difference of grid densities)
  - MMD^2 (Gaussian kernel, grid quadrature)

Power-law rates are estimated by log–log regression of metric vs. M_t.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import csv
import datetime
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import matplotlib.pyplot as plt

import os
import json
import csv
import datetime
import logging
from pathlib import Path

import numpy as np
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


# Reference: unbiased sampler for P0 (kept for comparison/debugging)
# def sample_true(n, rng):
#     if n <= 0:
#         return np.empty(0, dtype=float)
#     z = rng.random(n)
#     n1 = np.count_nonzero(z < w1)
#     n2 = n - n1
#     return np.concatenate([
#         rng.normal(mu1, sigma1, size=n1),
#         rng.normal(mu2, sigma2, size=n2),
#     ])


# Bias component Q (chosen to be clearly distinct from P0)
mu3, sigma3 = 2.0, 1.0

def sample_true(n, rng, bias=0.0):
    """
    Biased real-data sampler:

      With probability bias:   sample from Q = N(mu3, sigma3)
      Otherwise:               sample from P0 (mixture with weight w1)

    So the real distribution is:
      P_t = (1 - bias) * P0 + bias * Q
    """
    if n <= 0: return np.empty(0)
    u = rng.random(n)
    x = np.empty(n)
    x[u < bias] = rng.normal(mu3, sigma3, (u < bias).sum())
    x[(u >= bias) & (u < bias + (1-bias)*w1)] = rng.normal(mu1, sigma1, ((u >= bias) & (u < bias + (1-bias)*w1)).sum())
    x[u >= bias + (1-bias)*w1] = rng.normal(mu2, sigma2, (u >= bias + (1-bias)*w1).sum())
    return x


def true_pdf(x):
    """Target density f0 for P0."""
    return w1 * norm.pdf(x, mu1, sigma1) + (1 - w1) * norm.pdf(x, mu2, sigma2)


def true_cdf(x):
    """Target CDF F0 for P0."""
    return w1 * ndtr((x - mu1) / sigma1) + (1 - w1) * ndtr((x - mu2) / sigma2)


# Fixed grid for deterministic metric computation
left = min(mu1 - 6 * sigma1, mu2 - 6 * sigma2)
right = max(mu1 + 6 * sigma1, mu2 + 6 * sigma2)
m_grid = 200  # higher = more stable quadrature for W1/TV/MMD on grid
xg = np.linspace(left, right, m_grid)
dx = xg[1] - xg[0]
F_true_g = true_cdf(xg)
f_true_g = true_pdf(xg)


# ============================================================
# 2. Helpers: KDE / ECDF, CDF, W1, TV, MMD^2
# ============================================================
def silverman_bw(x):
    """Silverman bandwidth rule (unused if you manually set bandwidth schedule)."""
    s = np.std(x, ddof=1)
    n = len(x)
    if n <= 1 or s == 0:
        return 1.0
    return 1.06 * s * n ** (-1 / 5)


def kde_pdf(kde, x1d):
    """Evaluate KDE density on a 1D grid."""
    return np.exp(kde.score_samples(x1d.reshape(-1, 1)))


def cdf_from_pdf_on_grid(pdf_vals, dx, renormalize=True):
    """Compute CDF from grid PDF by trapezoidal integration (optionally renormalize)."""
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
    """W1 in 1D: ∫ |F_true(x) - F_model(x)| dx (approximated on grid)."""
    return np.trapezoid(np.abs(F_true - F_model), xg)


def empirical_cdf_on_grid(samples, xg):
    """Empirical CDF evaluated on xg: F_hat(x) = (1/n) Σ 1{Xi <= x}."""
    samples_sorted = np.sort(samples)
    n = len(samples_sorted)
    idx = np.searchsorted(samples_sorted, xg, side="right")
    return idx / n


def total_variation_from_pdfs(f_true, f_model, dx):
    """TV(P,Q) = 0.5 * ∫ |f_true(x) - f_model(x)| dx (approximated on grid)."""
    return 0.5 * np.trapezoid(np.abs(f_true - f_model), dx=dx)


def mmd2_from_pdfs(f_true, f_model, xg, sigma=0.5):
    """
    Grid-quadrature estimate of MMD^2 between two densities using a Gaussian kernel:
      k(x,y) = exp(-(x-y)^2 / (2 sigma^2))

    Returns MMD^2 (not the square root).
    """
    dx_local = xg[1] - xg[0]
    X = xg[:, None]
    K = np.exp(-(X - X.T) ** 2 / (2.0 * sigma ** 2))

    # ∫∫ f(x) f(x') k(x,x') dx dx' approximated with grid weights dx^2
    term11 = np.sum((f_true @ K) * f_true) * dx_local * dx_local
    term22 = np.sum((f_model @ K) * f_model) * dx_local * dx_local
    term12 = np.sum((f_true @ K) * f_model) * dx_local * dx_local
    return term11 + term22 - 2.0 * term12


# ============================================================
# 3. Recursive real + synthetic estimator (KDE or ECDF)
# ============================================================
def run_recursive_estimator(
    p_target=0.5,
    alpha=0.5,          # synthetic fraction used to set m2 from m1 (see computation below)
    m1=50,
    T=200,
    use_ecdf=False,
    h0=0.5,
    bias_rate=0.0,      # q in the bias schedule: bias_t = 0.1 * (t+5)^(-q)
):
    """
    Recursive scheme with biased real data and synthetic self-training.

    Real sampling at iteration t:
      P_t = (1 - bias_t) * P0 + bias_t * N(mu3, sigma3),
      bias_t = 0.1 * (t + 5)^(-bias_rate)   (if bias_rate > 0; else 0).

    Synthetic sampling at iteration t:
      Draw m2 samples from the current estimator (KDE sampling) or bootstrap (ECDF).

    Training set:
      Accumulate all past real + synthetic samples.

    Evaluation target:
      Always evaluate distances to P0 (unbiased target), not P_t.
    """

    rng = np.random.default_rng()  # no explicit seed

    # Set m2 from alpha and m1. For alpha=0, use m2=0.
    if alpha <= 0.0:
        m2 = 0
    else:
        m2 = int(round(alpha * m1 / (1 - alpha)))
    alpha_used = m2 / (m1 + m2) if (m1 + m2) > 0 else 0.0

    # Accumulated real and synthetic pools
    X_real = np.empty(0, dtype=float)
    X_synth = np.empty(0, dtype=float)

    # Time-series outputs
    Ms = np.empty(T, dtype=int)
    W1s = np.empty(T, dtype=float)
    bandwidths = np.empty(T, dtype=float)
    biases = np.empty(T, dtype=float)

    # Optional histories on the evaluation grid
    f_hat_history = []
    F_hat_history = []

    h0_init = None
    M1 = None

    for t in range(1, T + 1):
        idx = t - 1

        # Bias schedule for real sampling
        bias_t = (0.1) * (t + 5) ** (-bias_rate) if bias_rate > 0 else 0.0
        biases[idx] = bias_t

        # 1) Add new (biased) real samples
        x_new_real = sample_true(m1, rng, bias=bias_t)
        X_real = x_new_real if X_real.size == 0 else np.concatenate((X_real, x_new_real))

        # 2) Build accumulated training data
        X_train = np.concatenate((X_real, X_synth)) if X_synth.size > 0 else X_real
        M_t = X_train.size
        Ms[idx] = M_t

        if use_ecdf:
            # ECDF estimator: CDF on grid; density via numerical derivative (for TV/MMD on grid)
            F_hat_g = empirical_cdf_on_grid(X_train, xg)
            f_hat_g = np.gradient(F_hat_g, dx)
            bandwidths[idx] = 0.0

            # Synthetic data: bootstrap from X_train
            if m2 > 0:
                synth_samples = rng.choice(X_train, size=m2, replace=True)
                X_synth = synth_samples if X_synth.size == 0 else np.concatenate((X_synth, synth_samples))

        else:
            # KDE estimator with bandwidth schedule h_t = h0 * (M_t / M1)^(-p/2)
            X_train_2d = X_train.reshape(-1, 1)

            if h0_init is None:
                h0_init = h0
                M1 = M_t

            h_t = h0_init * (M_t / M1) ** (-p_target / 2.0)
            bandwidths[idx] = h_t

            kde = KernelDensity(kernel="gaussian", bandwidth=h_t).fit(X_train_2d)
            f_hat_g = kde_pdf(kde, xg)
            F_hat_g = cdf_from_pdf_on_grid(f_hat_g, dx)

            # Synthetic data: sample from KDE
            if m2 > 0:
                synth_samples = kde.sample(m2).ravel()
                X_synth = synth_samples if X_synth.size == 0 else np.concatenate((X_synth, synth_samples))

        # 3) Evaluate W1 to target P0 on the grid
        W1s[idx] = w1_from_cdfs_on_grid(F_true_g, F_hat_g, xg)

        # 4) Save histories
        f_hat_history.append(f_hat_g)
        F_hat_history.append(F_hat_g)

        if (t == 1) or (t % 50 == 0):
            tag = "ECDF" if use_ecdf else "KDE"
            extra = f"h={bandwidths[idx]:.4f} | " if not use_ecdf else ""
            print(
                f"[{tag} p={p_target:.2f}, alpha≈{alpha_used:.2f}, q={bias_rate:.2f}] "
                f"t={t:4d} | bias_t={bias_t:.4f} | M_t={M_t:6d} | {extra}W1={W1s[idx]:.6f}"
            )

    return {
        "M": Ms,
        "W1": W1s,
        "bandwidths": bandwidths,
        "biases": biases,
        "bias_rate": bias_rate,
        "p_target": p_target,
        "alpha": alpha_used,      # effective synthetic fraction implied by integer m2
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
    """
    Fit: log(vals) = a + b * log(M)
    Returns slope b (i.e., vals ~ M^b).
    """
    M = np.asarray(M)
    vals = np.asarray(vals)

    # Ensure we retain some tail points
    if drop_first >= len(M) - 5:
        drop_first = max(0, len(M) - 5)

    M_use = M[drop_first:]
    vals_use = vals[drop_first:]

    logM = np.log(M_use).reshape(-1, 1)
    logV = np.log(vals_use)
    reg = LinearRegression().fit(logM, logV)
    return reg.coef_[0], reg.intercept_

def estimate_power_law_with_loglog(M, vals, drop_first=10):
    """
    Fit: log(vals) ≈ a + b * log(M) + c * log log(M)

    Returns
    -------
    b : coefficient on log(M) (primary power-law exponent)
    c : coefficient on log log(M) (log-log correction term)
    a : intercept
    reg : fitted LinearRegression
    """
    M = np.asarray(M)
    vals = np.asarray(vals)

    # Ensure we retain some tail points
    if drop_first >= len(M) - 5:
        drop_first = max(0, len(M) - 5)

    M_use = M[drop_first:]
    vals_use = vals[drop_first:]

    # Need strictly positive quantities for logs
    logM = np.log(M_use)
    mask = (vals_use > 0) & (M_use > 1.0) & (logM > 0)

    if mask.sum() < 3:
        raise ValueError("Not enough valid points to fit log-log-log model.")

    M_use = M_use[mask]
    vals_use = vals_use[mask]
    logM = logM[mask]
    loglogM = np.log(logM)
    logV = np.log(vals_use)

    X = np.column_stack([logM, loglogM])
    reg = LinearRegression().fit(X, logV)

    b, c = reg.coef_
    a = reg.intercept_
    return b, c, a, reg



# ============================================================
# 5. JSON serialization helper
# ============================================================
def _make_json_serializable(obj):
    """Convert numpy scalars/arrays to plain Python types for json.dump."""
    import numpy as _np

    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    if isinstance(obj, (_np.float32, _np.float64)):
        return float(obj)
    if isinstance(obj, (_np.int32, _np.int64)):
        return int(obj)
    return obj




# ============================================================
# 6. Experiment driver (parallel reps; save all_results.json)
# ============================================================
# Assumes: run_recursive_estimator, estimate_power_law_slope,
#          total_variation_from_pdfs, mmd2_from_pdfs, f_true_g, xg, dx exist.

import numpy as np
import json
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import datetime

def run_one(alpha_real, q, p, rep, T, m1, h0, drop_first):
    """
    One replicate:
      - Convert alpha_real to the script's alpha_est = 1 - alpha_real
      - Run the recursive estimator (ECDF mode here)
      - Compute W1/TV/MMD^2 time series on the grid
      - Fit log–log slopes vs M_t on the tail
    """
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

    TVs = np.array([total_variation_from_pdfs(f_true_g, f_hist[t], dx) for t in range(T)], dtype=float)
    MMD2s = np.array([mmd2_from_pdfs(f_true_g, f_hist[t], xg, 0.5) for t in range(T)], dtype=float)

    # Critical correction (kept exactly as in your script)
    if np.isclose(p, alpha_real):
        logM = np.log(np.maximum(M, 2))
        W1 = W1 / logM
        TVs = TVs / logM
        MMD2s = MMD2s / (logM ** 2)

    sW1, _ = estimate_power_law_slope(M, W1, drop_first=drop_first)
    sTV, _ = estimate_power_law_slope(M, TVs, drop_first=drop_first)
    sM2, _ = estimate_power_law_slope(M, MMD2s, drop_first=drop_first)

    return dict(
        alpha_real=float(alpha_real),
        bias_rate=float(q),
        p_target=float(p),
        rep=int(rep),
        slope_W1=float(sW1),
        slope_TV=float(sTV),
        slope_MMD2=float(sM2),
    )

if __name__ == "__main__":
    p_values = [0.5]
    alphas = [0.25, 0.5, 0.75]
    bias_rates = [0.25, 0.5, 0.75]
    T, m1, n_reps, h0, drop_first = 2000, 25, 100, 0.5, 1500

    # Output directory (timestamped)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path("logs/KDE") / f"BCRT_quick_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = [(a, q, p, r, T, m1, h0, drop_first)
             for a in alphas for q in bias_rates for p in p_values for r in range(n_reps)]
    print(f"Total tasks={len(tasks)}")

    rep_results = []
    with ThreadPoolExecutor(max_workers=24) as ex:
        futs = [ex.submit(run_one, *t) for t in tasks]
        for f in as_completed(futs):
            rep_results.append(f.result())

    # Aggregate results by (alpha_real, q, p) to match your all_results schema
    all_results = []
    keys = sorted({(r["alpha_real"], r["bias_rate"], r["p_target"]) for r in rep_results})

    for alpha_real, q, p in keys:
        grp = [r for r in rep_results
               if r["alpha_real"] == alpha_real and r["bias_rate"] == q and r["p_target"] == p]

        slopes_W1 = np.array([r["slope_W1"] for r in grp], dtype=float)
        slopes_TV = np.array([r["slope_TV"] for r in grp], dtype=float)
        slopes_MMD2 = np.array([r["slope_MMD2"] for r in grp], dtype=float)

        # Per-replicate prints (kept; just reworded label text in comments above)
        print("\n" + "=" * 70)
        print(f"Running {len(grp)}×: p={p}, alpha_real={alpha_real:.2f}, q={q:.2f}")
        for r in sorted(grp, key=lambda z: z["rep"]):
            print(f"rep {r['rep'] + 1}: slope_W1={r['slope_W1']:.3f} slope_TV={r['slope_TV']:.3f} slope_MMD2={r['slope_MMD2']:.3f}")

        mean_W1 = float(slopes_W1.mean())
        std_W1  = float(slopes_W1.std(ddof=1)) if len(slopes_W1) > 1 else 0.0
        mean_TV = float(slopes_TV.mean())
        std_TV  = float(slopes_TV.std(ddof=1)) if len(slopes_TV) > 1 else 0.0
        mean_MMD2 = float(slopes_MMD2.mean())
        std_MMD2  = float(slopes_MMD2.std(ddof=1)) if len(slopes_MMD2) > 1 else 0.0

        # MMD slope from MMD^2 slope: slope(MMD) = 0.5 * slope(MMD^2)
        mean_MMD = mean_MMD2 / 2.0
        std_MMD  = std_MMD2 / 2.0

        theo = float(min(p, alpha_real, q))

        print(f"*** avg over {len(grp)} ***")
        print(f"mean slope W1={mean_W1:.3f} (std={std_W1:.3f}) | observed rate≈{-mean_W1:.3f}")
        print(f"mean slope TV={mean_TV:.3f} (std={std_TV:.3f})")
        print(f"mean slope MMD={mean_MMD:.3f} (std={std_MMD:.3f}) | mean slope MMD2={mean_MMD2:.3f}")
        print(f"theory rate min(p,alpha_real,q)={theo:.3f}")

        all_results.append(dict(
            alpha_real=float(alpha_real),
            bias_rate=float(q),
            p_target=float(p),

            slopes_W1=slopes_W1.tolist(),
            slopes_TV=slopes_TV.tolist(),
            slopes_MMD2=slopes_MMD2.tolist(),

            mean_slope_W1=float(mean_W1),
            mean_slope_TV=float(mean_TV),
            mean_slope_MMD=float(mean_MMD),
            mean_slope_MMD2=float(mean_MMD2),

            std_slope_W1=float(std_W1),
            std_slope_TV=float(std_TV),
            std_slope_MMD=float(std_MMD),
            std_slope_MMD2=float(std_MMD2),

            theoretical_rate=float(theo),
        ))

    # Save results (list-of-dicts JSON)
    with open(out_dir / "all_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved: {out_dir/'all_results.json'}")

    # Histogram of observed W1 rates (rate = -slope)
    plt.hist([-x for x in slopes_W1], bins=20)
    plt.xlabel("Observed rate (-slope_W1)")
    plt.ylabel("Count")
    plt.title("Histogram of observed W1 rates")
    plt.show()
