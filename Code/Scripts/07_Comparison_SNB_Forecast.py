import requests
import pandas as pd
from pyjstat import pyjstat
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import os
import re
warnings.filterwarnings('ignore')


# ----------------------------
# Paths
# ----------------------------
script_dir = Path(__file__).resolve().parent
file_path = script_dir.parent.parent / "Code" / "Data" / "Raw_Data"
csv_path = file_path / "SNB_comparison.csv"

# ----------------------------
# Robust load: skip SNB cube metadata until the real header row
# ----------------------------
with open(csv_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

start = None
for i, line in enumerate(lines):
    if line.strip().startswith('"Date";"D0";"D1";"Value"'):
        start = i
        break

if start is None:
    raise ValueError('Could not find the SNB data header row: "Date";"D0";"D1";"Value"')

df = pd.read_csv(
    csv_path,
    sep=";",
    quotechar='"',
    skiprows=start,      # start reading at the actual header row
    dtype=str
)

# Rename to consistent column names
df = df.rename(columns={"Date": "target_q", "D0": "series", "D1": "kind", "Value": "value"})

# Basic cleaning
df["target_q"] = df["target_q"].astype(str).str.strip()
df["series"]   = df["series"].astype(str).str.strip()
df["kind"]     = df["kind"].astype(str).str.strip()
df["value"]    = pd.to_numeric(df["value"], errors="coerce")

# Drop any empty rows just in case
df = df.dropna(subset=["target_q", "series", "kind"], how="any").copy()

# Keep only proper quarter labels and proper series labels (extra safety)
df = df[df["target_q"].str.match(r"^\d{4}-Q[1-4]$", na=False)].copy()
df = df[df["series"].str.match(r"^[MJSD]\d{4}", na=False)].copy()

# ----------------------------
# Parse vintage and scenario from series code
# ----------------------------
month_map = {"M": 3, "J": 6, "S": 9, "D": 12}
m = df["series"].str.extract(r"^(?P<mon>[MJSD])(?P<year>\d{4})(?P<scenario>.*)$")
df = df.join(m)

df["vintage_month"] = df["mon"].map(month_map)
df["vintage_year"]  = pd.to_numeric(df["year"], errors="coerce")

# Vintage quarter (publication quarter)
df["vintage_q"] = pd.PeriodIndex(
    pd.to_datetime(dict(year=df["vintage_year"], month=df["vintage_month"], day=1)),
    freq="Q"
)

# Target quarter
df["target_qp"] = pd.PeriodIndex(df["target_q"], freq="Q")

# Forecast vs observed (as encoded by SNB cube)
df["what"] = df["kind"].map({"P": "forecast", "BI": "observed"})
df = df[df["what"].notna()].copy()

# ----------------------------
# Build panel: (vintage_q, target_qp, scenario) -> forecast / observed
# ----------------------------
panel = (
    df.pivot_table(
        index=["vintage_q", "target_qp", "scenario"],
        columns="what",
        values="value",
        aggfunc="first"
    )
    .reset_index()
)

# ----------------------------
# Choose the "active" scenario per vintage
# (recommended: restrict to PL* scenarios that clearly encode a policy-rate assumption)
# ----------------------------
panel_pl = panel[panel["scenario"].astype(str).str.startswith("PL")].copy()

# Count non-missing forecast entries per vintage & scenario
counts = (
    panel_pl
    .groupby(["vintage_q", "scenario"])["forecast"]
    .apply(lambda s: s.notna().sum())
    .reset_index(name="n_forecasts")
)

# Choose scenario with max forecasts per vintage
chosen = counts.loc[counts.groupby("vintage_q")["n_forecasts"].idxmax()].copy()

# Keep only chosen scenario
panel_snb = panel_pl.merge(
    chosen[["vintage_q", "scenario"]],
    on=["vintage_q", "scenario"],
    how="inner"
)

# Sanity check: one scenario per vintage
if not (panel_snb.groupby("vintage_q")["scenario"].nunique() <= 1).all():
    raise RuntimeError("More than one scenario per vintage remained after selection. Inspect 'panel_snb'.")

# ----------------------------
# Compute horizons and extract 3/6/9/12 months (1/2/3/4 quarters ahead)
# ----------------------------
panel_snb["h_q"] = panel_snb["target_qp"].astype("int64") - panel_snb["vintage_q"].astype("int64")
panel_snb["h_months"] = panel_snb["h_q"] * 3

snb_36912 = panel_snb[panel_snb["h_q"].isin([1, 2, 3, 4])].copy()
snb_36912 = snb_36912.sort_values(["vintage_q", "h_q", "target_qp"]).reset_index(drop=True)

# Final benchmark table: what you'll merge on later (vintage_q, h_months)
snb_path = snb_36912[["vintage_q", "h_months", "target_qp", "scenario", "forecast", "observed"]].copy()
# ------------------------------------------------------------
# Build realized quarterly inflation series from BI (observed)
# Take the latest available observed value for each target quarter
# ------------------------------------------------------------
obs_raw = panel.copy()

# Keep rows where observed exists
obs_raw = obs_raw[obs_raw["observed"].notna()].copy()

# Sort by vintage, then for each target quarter take the last observed value available
obs_raw = obs_raw.sort_values(["target_qp", "vintage_q"])

realized_q = (obs_raw
              .groupby("target_qp", as_index=False)
              .tail(1)[["target_qp", "observed"]]
              .rename(columns={"observed": "realized"}))

# Merge realized values into your snb_path (which is future-horizon forecasts)
snb_path = snb_path.merge(realized_q, on="target_qp", how="left")

print("\nCheck realized merge:")
print(snb_path[["vintage_q", "h_months", "target_qp", "forecast", "realized"]].head(20))

print("\nHow many realized values missing?")
print(snb_path["realized"].isna().mean())

# ----------------------------
# Output sanity prints
# ----------------------------
print("Loaded rows (clean):", len(df))
print("Panel rows:", len(panel))
print("SNB benchmark rows (h=3,6,9,12):", len(snb_path))
print("\nUnique scenarios retained (should be small):")
print(snb_path["scenario"].value_counts().head(10))

print("\nPreview snb_path:")
print(snb_path.head(20))




import numpy as np
import pandas as pd
from scipy.stats import norm

# ============================================================
# Helpers
# ============================================================

def rmse(y, yhat):
    m = np.isfinite(y) & np.isfinite(yhat)
    return np.sqrt(np.mean((y[m] - yhat[m])**2)) if m.any() else np.nan

def mae(y, yhat):
    m = np.isfinite(y) & np.isfinite(yhat)
    return np.mean(np.abs(y[m] - yhat[m])) if m.any() else np.nan

def interval_score(y, l, u, coverage):
    """
    Gneiting & Raftery interval score.
    coverage = 0.90 -> alpha = 0.10
    score = (u-l) + (2/alpha)*(l-y) if y<l + (2/alpha)*(y-u) if y>u
    """
    alpha = 1.0 - coverage
    y = np.asarray(y)
    l = np.asarray(l)
    u = np.asarray(u)

    score = (u - l).copy()
    below = y < l
    above = y > u
    score[below] += (2.0 / alpha) * (l[below] - y[below])
    score[above] += (2.0 / alpha) * (y[above] - u[above])
    return score

def coverage_rate(y, l, u):
    m = np.isfinite(y) & np.isfinite(l) & np.isfinite(u)
    if not m.any():
        return np.nan
    return np.mean((y[m] >= l[m]) & (y[m] <= u[m]))

def mean_width(l, u):
    m = np.isfinite(l) & np.isfinite(u)
    return np.mean(u[m] - l[m]) if m.any() else np.nan

def prob_below_from_quantiles(x, q_levels, q_values):
    """
    Approximate P(Y <= x) given a monotone quantile function.
    q_levels: array of probabilities (e.g., [0.01,...,0.99])
    q_values: array of corresponding quantile values for one forecast
    Uses linear interpolation on the inverse CDF.
    """
    q_levels = np.asarray(q_levels, dtype=float)
    q_values = np.asarray(q_values, dtype=float)

    m = np.isfinite(q_levels) & np.isfinite(q_values)
    q_levels = q_levels[m]
    q_values = q_values[m]
    if len(q_levels) < 2:
        return np.nan

    # ensure sorted by quantile value (should already be monotone, but enforce)
    order = np.argsort(q_values)
    q_values = q_values[order]
    q_levels = q_levels[order]

    if x <= q_values[0]:
        return float(q_levels[0])
    if x >= q_values[-1]:
        return float(q_levels[-1])

    return float(np.interp(x, q_values, q_levels))

def add_snb_gaussian_intervals(df, coverage_levels=(0.5, 0.7, 0.9), min_obs=20, expanding=True):
    """
    Build SNB 'Gaussian' intervals around SNB point forecast using historical SNB errors by horizon.
    - expanding=True: expanding window std
    - else: rolling window (set rolling size below if you extend)
    Returns df with columns like snb_l90, snb_u90, ...
    """
    out = df.copy()
    out = out.sort_values(["h_months", "vintage_q"]).reset_index(drop=True)

    # SNB forecast error
    out["snb_err"] = out["forecast"] - out["realized"]

    # estimate sigma per horizon over time
    sigmas = []
    for h, g in out.groupby("h_months", sort=False):
        g = g.copy().sort_values("vintage_q")
        errs = g["snb_err"].to_numpy(dtype=float)

        sigma_t = np.full(len(g), np.nan, dtype=float)
        for i in range(len(g)):
            hist = errs[:i]  # only past errors (strictly before current vintage)
            hist = hist[np.isfinite(hist)]
            if len(hist) >= min_obs:
                sigma_t[i] = np.std(hist, ddof=1)
        g["snb_sigma"] = sigma_t
        sigmas.append(g)

    out = pd.concat(sigmas, axis=0).sort_values(["h_months", "vintage_q"]).reset_index(drop=True)

    # build intervals
    for cov in coverage_levels:
        alpha = 1.0 - cov
        z = norm.ppf(1.0 - alpha/2.0)
        out[f"snb_l{int(cov*100)}"] = out["forecast"] - z*out["snb_sigma"]
        out[f"snb_u{int(cov*100)}"] = out["forecast"] + z*out["snb_sigma"]

    return out

# ============================================================
# MAIN: compute metrics
# ============================================================

def compute_snb_benchmark_metrics(
    snb_path: pd.DataFrame,
    model_fc: pd.DataFrame,
    quantile_cols: dict,
    dense_q_levels: list | None = None,
    dense_q_prefix: str = "q",
    coverage_levels=(0.5, 0.7, 0.9),
    tail_thresholds=(0.0, 2.0),
):
    """
    snb_path: must have ['vintage_q','h_months','forecast','realized']
    model_fc: must have ['vintage_q','h_months'] plus quantiles specified in quantile_cols
              quantile_cols example for 90% interval:
              {'q50':'q50', 0.05:'q05', 0.95:'q95', 0.15:'q15', 0.85:'q85', 0.25:'q25', 0.75:'q75'}
    dense_q_levels: optional list like [0.01,...,0.99] if you have dense quantiles for better tail probs
    dense_q_prefix: column prefix if dense quantiles named 'q01','q02',... (as strings)

    Returns:
      merged_df, metrics_table (by horizon and overall)
    """

    # --- ensure Period dtype for vintage_q if needed
    def to_period_q(x):
        if isinstance(x.dtype, pd.PeriodDtype):
            return x
        return pd.PeriodIndex(x, freq="Q")

    snb = snb_path.copy()
    mdl = model_fc.copy()

    snb["vintage_q"] = to_period_q(snb["vintage_q"])
    mdl["vintage_q"] = to_period_q(mdl["vintage_q"])

    # --- merge
    df = snb.merge(mdl, on=["vintage_q", "h_months"], how="inner")
    df = df.dropna(subset=["realized", "forecast"]).copy()

    # --- model point (median)
    q50_col = quantile_cols.get("q50", "q50")
    df["model_point"] = df[q50_col]

    # --- point accuracy
    # (Compute per horizon and overall)
    rows = []
    for h, g in df.groupby("h_months"):
        rows.append({
            "scope": f"h={h}m",
            "n": len(g),
            "RMSE_model": rmse(g["realized"].values, g["model_point"].values),
            "RMSE_SNB": rmse(g["realized"].values, g["forecast"].values),
            "MAE_model": mae(g["realized"].values, g["model_point"].values),
            "MAE_SNB": mae(g["realized"].values, g["forecast"].values),
        })
    rows.append({
        "scope": "overall",
        "n": len(df),
        "RMSE_model": rmse(df["realized"].values, df["model_point"].values),
        "RMSE_SNB": rmse(df["realized"].values, df["forecast"].values),
        "MAE_model": mae(df["realized"].values, df["model_point"].values),
        "MAE_SNB": mae(df["realized"].values, df["forecast"].values),
    })

    # --- interval metrics for your model
    for cov in coverage_levels:
        lq = round((1.0 - cov)/2.0, 2)
        uq = round(1.0 - (1.0 - cov)/2.0, 2)

        lcol = quantile_cols.get(lq, None)
        ucol = quantile_cols.get(uq, None)
        if lcol is None or ucol is None:
            # skip if you don't have those endpoints
            continue

        df[f"model_l{int(cov*100)}"] = df[lcol]
        df[f"model_u{int(cov*100)}"] = df[ucol]

        # interval score per row
        df[f"model_IS{int(cov*100)}"] = interval_score(
            df["realized"].values,
            df[f"model_l{int(cov*100)}"].values,
            df[f"model_u{int(cov*100)}"].values,
            coverage=cov
        )

    # --- SNB Gaussian benchmark intervals (constructed)
    df = add_snb_gaussian_intervals(df, coverage_levels=coverage_levels, min_obs=20)

    for cov in coverage_levels:
        lcol = f"snb_l{int(cov*100)}"
        ucol = f"snb_u{int(cov*100)}"
        if lcol in df.columns and ucol in df.columns:
            df[f"snb_IS{int(cov*100)}"] = interval_score(
                df["realized"].values,
                df[lcol].values,
                df[ucol].values,
                coverage=cov
            )

    # --- tail probabilities from quantiles
    # Prefer dense quantiles if provided; otherwise approximate from available endpoints.
    lo_thr, hi_thr = tail_thresholds

    if dense_q_levels is not None:
        # expect columns like q01,q02,... or similar; adapt if needed
        dense_cols = []
        for a in dense_q_levels:
            # e.g., 0.01 -> "q01"
            dense_cols.append(f"{dense_q_prefix}{int(round(a*100)):02d}")

        missing = [c for c in dense_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Dense quantile columns missing: {missing[:10]} ...")

        q_levels = np.array(dense_q_levels, dtype=float)

        p_below = []
        p_above = []
        p_in = []
        for _, r in df.iterrows():
            q_vals = r[dense_cols].to_numpy(dtype=float)
            p_lo = prob_below_from_quantiles(lo_thr, q_levels, q_vals)
            p_hi = prob_below_from_quantiles(hi_thr, q_levels, q_vals)
            p_below.append(p_lo)
            p_above.append(1.0 - p_hi if np.isfinite(p_hi) else np.nan)
            p_in.append((p_hi - p_lo) if (np.isfinite(p_hi) and np.isfinite(p_lo)) else np.nan)

        df["p_below0"] = p_below
        df["p_above2"] = p_above
        df["p_in_0_2"] = p_in

    else:
        # Use whatever quantiles you have (less accurate, but works)
        # Build arrays from all numeric quantiles provided in quantile_cols (excluding 'q50' key).
        q_pairs = [(k, v) for k, v in quantile_cols.items() if isinstance(k, float)]
        q_pairs = sorted(q_pairs, key=lambda x: x[0])
        q_levels = np.array([k for k, _ in q_pairs], dtype=float)
        q_cols = [v for _, v in q_pairs]
        if len(q_cols) < 3:
            raise ValueError("Not enough quantiles to approximate tail probs. Provide dense_q_levels or more quantiles.")

        p_below, p_above, p_in = [], [], []
        for _, r in df.iterrows():
            q_vals = r[q_cols].to_numpy(dtype=float)
            p_lo = prob_below_from_quantiles(lo_thr, q_levels, q_vals)
            p_hi = prob_below_from_quantiles(hi_thr, q_levels, q_vals)
            p_below.append(p_lo)
            p_above.append(1.0 - p_hi if np.isfinite(p_hi) else np.nan)
            p_in.append((p_hi - p_lo) if (np.isfinite(p_hi) and np.isfinite(p_lo)) else np.nan)

        df["p_below0"] = p_below
        df["p_above2"] = p_above
        df["p_in_0_2"] = p_in

    # --- build metrics table
    metrics = []

    def add_interval_metrics(scope, g, prefix, cov):
        l = g[f"{prefix}_l{int(cov*100)}"].values
        u = g[f"{prefix}_u{int(cov*100)}"].values
        y = g["realized"].values
        return {
            f"cov{int(cov*100)}_{prefix}": coverage_rate(y, l, u),
            f"width{int(cov*100)}_{prefix}": mean_width(l, u),
            f"IS{int(cov*100)}_{prefix}": np.nanmean(g[f"{prefix}_IS{int(cov*100)}"].values) if f"{prefix}_IS{int(cov*100)}" in g.columns else np.nan,
        }

    for h, g in df.groupby("h_months"):
        row = {"scope": f"h={h}m", "n": len(g)}
        row.update({
            "RMSE_model": rmse(g["realized"].values, g["model_point"].values),
            "RMSE_SNB": rmse(g["realized"].values, g["forecast"].values),
            "MAE_model": mae(g["realized"].values, g["model_point"].values),
            "MAE_SNB": mae(g["realized"].values, g["forecast"].values),
            "mean_p_below0": np.nanmean(g["p_below0"].values),
            "mean_p_above2": np.nanmean(g["p_above2"].values),
            "mean_p_in_0_2": np.nanmean(g["p_in_0_2"].values),
        })

        for cov in coverage_levels:
            if f"model_l{int(cov*100)}" in g.columns and f"model_u{int(cov*100)}" in g.columns:
                row.update(add_interval_metrics("x", g, "model", cov))
            if f"snb_l{int(cov*100)}" in g.columns and f"snb_u{int(cov*100)}" in g.columns:
                row.update(add_interval_metrics("x", g, "snb", cov))

        metrics.append(row)

    # overall
    g = df
    row = {"scope": "overall", "n": len(g)}
    row.update({
        "RMSE_model": rmse(g["realized"].values, g["model_point"].values),
        "RMSE_SNB": rmse(g["realized"].values, g["forecast"].values),
        "MAE_model": mae(g["realized"].values, g["model_point"].values),
        "MAE_SNB": mae(g["realized"].values, g["forecast"].values),
        "mean_p_below0": np.nanmean(g["p_below0"].values),
        "mean_p_above2": np.nanmean(g["p_above2"].values),
        "mean_p_in_0_2": np.nanmean(g["p_in_0_2"].values),
    })
    for cov in coverage_levels:
        if f"model_l{int(cov*100)}" in g.columns and f"model_u{int(cov*100)}" in g.columns:
            row.update(add_interval_metrics("x", g, "model", cov))
        if f"snb_l{int(cov*100)}" in g.columns and f"snb_u{int(cov*100)}" in g.columns:
            row.update(add_interval_metrics("x", g, "snb", cov))
    metrics.append(row)

    metrics_df = pd.DataFrame(metrics)

    return df, metrics_df


# ============================================================
# HOW TO USE (EDIT THESE MAPPINGS TO YOUR COLUMN NAMES)
# ============================================================

# Example mapping if your model forecast df has columns like:
# q05, q15, q25, q50, q75, q85, q95
QUANTILE_COLS = {
    "q50": "q50",
    0.05: "q05", 0.95: "q95",   # 90%
    0.15: "q15", 0.85: "q85",   # 70%
    0.25: "q25", 0.75: "q75",   # 50%
}

# If you have dense quantiles q01..q99, set:
# dense_levels = [i/100 for i in range(1, 100)]
dense_levels = None  # set to list if available
dense_prefix = "q"   # if columns are q01,q02,...

# Run:
# merged_df, metrics_table = compute_snb_benchmark_metrics(snb_path, model_fc, QUANTILE_COLS, dense_levels, dense_prefix)