from __future__ import annotations

import os
import re
import glob
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm


# -----------------------------
# CONFIG
# -----------------------------
folders = [
    "Results/Data_experiments_benchmark",
    "Results/Data_experiments_bvar",
]

# Prefer this Track A loss for GW(CPA). If missing, fallback will be used.
PREFERRED_GW_LOSS = "NegLogS_direct"          # optional
FALLBACK_GW_LOSS = "CRPS_direct_parametric"  # required if preferred missing

# Track C violations column for LR tests (90% band => alpha=0.10)
VIOL_COL = "Violation90_YoY_timesafe"
ALPHA = 0.10

# PIT column for independence test
PIT_COL = "PIT_direct"

# Ljung–Box lags for PIT independence
PIT_LB_LAGS = 12

# Newey–West lags for GW(CPA) mean test (common choice)
GW_NW_LAGS = 4

# Output
save_path = "Scripts/Plots_and_Tables/06b_Statistical_tests/06b_Statistical_tests.csv"


# -----------------------------
# HELPERS: filename parsing
# -----------------------------
H_RE = re.compile(r"(?<!\d)(\d{1,2})m(?!\d)", re.IGNORECASE)

def parse_target_and_horizon(filename: str) -> Tuple[str, Optional[int]]:
    lname = filename.lower()
    if "headline" in lname:
        target = "Headline"
    elif "core" in lname:
        target = "Core"
    else:
        target = "Unknown"

    mh = H_RE.search(filename)
    horizon = int(mh.group(1)) if mh else None
    return target, horizon


# -----------------------------
# GW(CPA): HAC mean test of loss differential
# -----------------------------
def newey_west_variance_of_mean(x: np.ndarray, L: int) -> float:
    """
    Newey–West estimator for Var(mean(x)).
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    T = x.size
    if T <= 1:
        return np.nan

    x = x - x.mean()
    gamma0 = float(np.dot(x, x) / T)
    var = gamma0

    max_l = min(L, T - 1)
    for l in range(1, max_l + 1):
        w = 1.0 - l / (max_l + 1.0)
        gam = float(np.dot(x[l:], x[:-l]) / T)
        var += 2.0 * w * gam

    return var / T


def gw_cpa_pvalue(loss_model: np.ndarray, loss_bench: np.ndarray, nw_lags: int = 4) -> Tuple[float, float, int]:
    """
    CPA special case: test E[d_t] = 0 using HAC (Newey–West),
    where d_t = loss_model - loss_bench.
    Returns (stat, pvalue, n).

    Interpretation:
    - If mean(d) < 0, model has lower loss than benchmark.
    - p-value is two-sided for E[d]=0.
    """
    from scipy.stats import norm as znorm

    loss_model = np.asarray(loss_model, float)
    loss_bench = np.asarray(loss_bench, float)

    mask = np.isfinite(loss_model) & np.isfinite(loss_bench)
    d = (loss_model[mask] - loss_bench[mask]).astype(float)
    n = int(d.size)
    if n < 20:
        return np.nan, np.nan, n

    v = newey_west_variance_of_mean(d, L=nw_lags)
    if not np.isfinite(v) or v <= 0:
        return np.nan, np.nan, n

    stat = float(d.mean() / np.sqrt(v))
    pval = float(2.0 * (1.0 - znorm.cdf(abs(stat))))
    return stat, pval, n


# -----------------------------
# Christoffersen LR tests: LRuc, LRind, LRcc
# -----------------------------
def _ll_binom(p: float, k: int, n: int) -> float:
    p = float(np.clip(p, 1e-12, 1 - 1e-12))
    return k * np.log(p) + (n - k) * np.log(1 - p)


def christoffersen_lr_pvalues(I: np.ndarray, alpha: float = 0.10) -> Dict[str, float]:
    """
    I_t in {0,1}, 1 indicates violation. alpha nominal violation probability.

    Returns:
      pi_hat, LRuc, p_LRuc, LRind, p_LRind, LRcc, p_LRcc, n
    """
    I = np.asarray(I, float)
    I = I[np.isfinite(I)].astype(int)
    T = int(I.size)

    out = {
        "n_cov": T,
        "pi_hat": np.nan,
        "p_LRuc": np.nan,
        "p_LRind": np.nan,
        "p_LRcc": np.nan,
    }
    if T < 30:
        if T > 0:
            out["pi_hat"] = float(I.mean())
        return out

    x = int(I.sum())
    pi_hat = x / T
    out["pi_hat"] = float(pi_hat)

    # LRuc
    LRuc = -2.0 * (_ll_binom(alpha, x, T) - _ll_binom(pi_hat, x, T))
    out["p_LRuc"] = float(1.0 - chi2.cdf(LRuc, df=1))

    # LRind transitions
    I0, I1 = I[:-1], I[1:]
    n00 = int(((I0 == 0) & (I1 == 0)).sum())
    n01 = int(((I0 == 0) & (I1 == 1)).sum())
    n10 = int(((I0 == 1) & (I1 == 0)).sum())
    n11 = int(((I0 == 1) & (I1 == 1)).sum())

    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi1 = (n01 + n11) / (n00 + n01 + n10 + n11)

    ll_markov = _ll_binom(pi01, n01, n00 + n01) + _ll_binom(pi11, n11, n10 + n11)
    ll_iid = _ll_binom(pi1, n01 + n11, n00 + n01 + n10 + n11)

    LRind = -2.0 * (ll_iid - ll_markov)
    out["p_LRind"] = float(1.0 - chi2.cdf(LRind, df=1))

    LRcc = LRuc + LRind
    out["p_LRcc"] = float(1.0 - chi2.cdf(LRcc, df=2))

    return out


# -----------------------------
# PIT independence: Ljung–Box on z = Phi^{-1}(PIT)
# -----------------------------
def pit_ljung_box_pvalue(pit: np.ndarray, lags: int = 12) -> Dict[str, float]:
    """
    Tests serial independence of PITs via Ljung–Box on z = Phi^{-1}(PIT).
    Returns n_pit, p_LB, LB_stat.
    """
    pit = np.asarray(pit, float)
    pit = pit[np.isfinite(pit)]
    pit = np.clip(pit, 1e-6, 1 - 1e-6)
    n = int(pit.size)

    out = {"n_pit": n, "p_PIT_LB": np.nan, "LB_stat": np.nan, "PIT_mean": np.nan}
    if n < 30:
        if n > 0:
            out["PIT_mean"] = float(pit.mean())
        return out

    out["PIT_mean"] = float(pit.mean())
    z = norm.ppf(pit)
    z = z - z.mean()

    denom = float(np.dot(z, z))
    if not np.isfinite(denom) or denom <= 0:
        return out

    max_l = min(lags, n - 1)
    if max_l < 1:
        return out

    ac2_sum = 0.0
    for k in range(1, max_l + 1):
        num = float(np.dot(z[k:], z[:-k]))
        rho = num / denom
        ac2_sum += (rho * rho) / (n - k)

    Q = n * (n + 2.0) * ac2_sum
    out["LB_stat"] = float(Q)
    out["p_PIT_LB"] = float(1.0 - chi2.cdf(Q, df=max_l))
    return out


# -----------------------------
# MAIN: build table
# -----------------------------
def main() -> None:
    # Collect all csvs and group by (Target, Horizon, ModelKey(folder))
    by_target_horizon: Dict[Tuple[str, int], Dict[str, pd.DataFrame]] = {}

    for folder in folders:
        model_key = folder.split("/")[-1]
        file_paths = glob.glob(os.path.join(folder, "*.csv"))

        for path in file_paths:
            file_name = os.path.basename(path)
            target, horizon = parse_target_and_horizon(file_name)
            if target == "Unknown" or horizon is None:
                continue

            df = pd.read_csv(path)

            # Keep only relevant columns (but don't crash if extra columns exist)
            by_target_horizon.setdefault((target, horizon), {})[model_key] = df

    if not by_target_horizon:
        raise RuntimeError("No result files found / parsed. Check folders and filename conventions.")

    # Choose benchmark model key = the benchmark folder name
    bench_key = folders[0].split("/")[-1]
    if all(bench_key not in d for d in by_target_horizon.values()):
        raise RuntimeError(f"Benchmark key '{bench_key}' not found in parsed data. Check folders[0].")

    rows: List[Dict[str, object]] = []

    for (target, horizon), model_map in sorted(by_target_horizon.items(), key=lambda x: (x[0][0], x[0][1])):
        if bench_key not in model_map:
            # can't do GW without benchmark for this target/horizon
            continue

        df_bench = model_map[bench_key]

        # Choose GW loss column
        gw_loss_col = PREFERRED_GW_LOSS if PREFERRED_GW_LOSS in df_bench.columns else FALLBACK_GW_LOSS
        if gw_loss_col not in df_bench.columns:
            # skip GW if no loss available
            gw_loss_col = None

        # Benchmark loss series
        bench_loss = pd.to_numeric(df_bench[gw_loss_col], errors="coerce").values if gw_loss_col else None

        for model_key, df in model_map.items():
            # GW p-value (model vs benchmark). For benchmark itself, leave NaN.
            gw_p = np.nan
            gw_stat = np.nan
            gw_n = np.nan
            mean_loss_diff = np.nan

            if model_key != bench_key and gw_loss_col and gw_loss_col in df.columns:
                model_loss = pd.to_numeric(df[gw_loss_col], errors="coerce").values

                # Align by Target_date if available; otherwise align by row index
                if "Target_date" in df.columns and "Target_date" in df_bench.columns:
                    jb = df_bench[["Target_date", gw_loss_col]].copy()
                    jm = df[["Target_date", gw_loss_col]].copy()
                    jb["Target_date"] = pd.to_datetime(jb["Target_date"], errors="coerce")
                    jm["Target_date"] = pd.to_datetime(jm["Target_date"], errors="coerce")
                    merged = jb.merge(jm, on="Target_date", how="inner", suffixes=("_b", "_m"))
                    b = pd.to_numeric(merged[f"{gw_loss_col}_b"], errors="coerce").values
                    m = pd.to_numeric(merged[f"{gw_loss_col}_m"], errors="coerce").values
                else:
                    # fall back: elementwise alignment
                    b = bench_loss
                    m = model_loss

                # compute
                gw_stat, gw_p, n = gw_cpa_pvalue(m, b, nw_lags=GW_NW_LAGS)
                gw_n = n
                # mean loss diff
                mask = np.isfinite(m) & np.isfinite(b)
                mean_loss_diff = float(np.mean(m[mask] - b[mask])) if mask.any() else np.nan

            # Coverage LR tests (Track C)
            cov = {}
            if VIOL_COL in df.columns:
                I = pd.to_numeric(df[VIOL_COL], errors="coerce").values
                cov = christoffersen_lr_pvalues(I, alpha=ALPHA)

            # PIT independence (Track A)
            pit_res = {}
            if PIT_COL in df.columns:
                pit = pd.to_numeric(df[PIT_COL], errors="coerce").values
                pit_res = pit_ljung_box_pvalue(pit, lags=PIT_LB_LAGS)

            rows.append({
                "Model": model_key,
                "Benchmark": bench_key,
                "Target": target,
                "Horizon_m": horizon,

                # GW(CPA)
                "GW_loss_col": gw_loss_col if gw_loss_col else "",
                "GW_mean_loss_diff": mean_loss_diff,  # model - benchmark (negative => model better)
                "GW_stat": gw_stat,
                "GW_p": gw_p,
                "GW_n": gw_n,

                # LR tests (Track C)
                "pi_hat": cov.get("pi_hat", np.nan),
                "p_LRuc": cov.get("p_LRuc", np.nan),
                "p_LRind": cov.get("p_LRind", np.nan),
                "p_LRcc": cov.get("p_LRcc", np.nan),
                "n_cov": cov.get("n_cov", np.nan),

                # PIT independence
                "PIT_mean": pit_res.get("PIT_mean", np.nan),
                "p_PIT_LB": pit_res.get("p_PIT_LB", np.nan),
                "LB_stat": pit_res.get("LB_stat", np.nan),
                "n_pit": pit_res.get("n_pit", np.nan),
            })

    tests_df = pd.DataFrame(rows).sort_values(["Target", "Horizon_m", "Model"]).reset_index(drop=True)

    # Ensure output directory exists
    out_dir = os.path.dirname(save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Print + save
    print(tests_df.to_string(index=False))
    tests_df.to_csv(save_path, index=False)
    print(f"\n[INFO] Saved post-hoc tests table to: {save_path}")


if __name__ == "__main__":
    main()