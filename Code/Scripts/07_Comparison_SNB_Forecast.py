import pandas as pd
import numpy as np
from pathlib import Path
import re
import warnings
import matplotlib.dates as mdates
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent

# --- paths (adapt to your repo structure) ---
RAW_DIR = SCRIPT_DIR.parent.parent / "Code" / "Data" / "Raw_Data"
RES_DIR = SCRIPT_DIR.parent.parent / "Code" / "Results" / "Data_experiments_qrf"  # adapt

SNB_CSV = RAW_DIR / "SNB_comparison.csv"

# Your best model files (edit names as needed)
# If later you want to run for all models, add another dict and loop.
MODEL_NAME = "QRF_Default_PCA_Headline"
MODEL_FILES = {
    3: RES_DIR / "QRF_Default_PCA_Headline_3m.csv",
    6: RES_DIR / "QRF_Default_PCA_Headline_6m.csv",
    9: RES_DIR / "QRF_Default_PCA_Headline_9m.csv",
    12: RES_DIR / "QRF_Default_PCA_Headline_12m.csv",
}

USE_TIMESAFE = True  # recommended for honesty about information set
PLOT_DIR = SCRIPT_DIR / "Plots_and_Tables"/"07_Comparison_SNB_Forecast"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Which intervals to check
COVERAGES = [0.68, 0.90]  # align to your q16/q84 and q05/q95
TAIL_THRESHOLDS = [0.0, 2.0]  # deflation and above-target thresholds


# ============================================================
# 1) LOAD + PARSE SNB CUBE EXPORT (robust skip of metadata)
# ============================================================
def load_snb_cube(csv_path: Path) -> pd.DataFrame:
    with open(csv_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith('"Date";"D0";"D1";"Value"'):
            start = i
            break
    if start is None:
        raise ValueError('Header row not found: "Date";"D0";"D1";"Value"')

    df = pd.read_csv(csv_path, sep=";", quotechar='"', skiprows=start, dtype=str)
    df = df.rename(columns={"Date": "target_q", "D0": "series", "D1": "kind", "Value": "value"})

    # Clean
    for c in ["target_q", "series", "kind"]:
        df[c] = df[c].astype(str).str.strip()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # keep only valid rows
    df = df[df["target_q"].str.match(r"^\d{4}-Q[1-4]$", na=False)].copy()
    df = df[df["series"].str.match(r"^[MJSD]\d{4}", na=False)].copy()
    df = df[df["kind"].isin(["P", "BI"])].copy()  # P=forecast, BI=observed (in cube)
    return df


def snb_build_path(df: pd.DataFrame) -> pd.DataFrame:
    # series: e.g. M2012PL000P -> publication month letter + year + scenario
    month_map = {"M": 3, "J": 6, "S": 9, "D": 12}
    m = df["series"].str.extract(r"^(?P<mon>[MJSD])(?P<year>\d{4})(?P<scenario>.*)$")
    df = df.join(m)

    df["vintage_month"] = df["mon"].map(month_map)
    df["vintage_year"] = pd.to_numeric(df["year"], errors="coerce")

    df["vintage_q"] = pd.PeriodIndex(
        pd.to_datetime(dict(year=df["vintage_year"], month=df["vintage_month"], day=1)),
        freq="Q"
    )
    df["target_qp"] = pd.PeriodIndex(df["target_q"], freq="Q")

    df["what"] = df["kind"].map({"P": "forecast", "BI": "observed"})
    df = df[df["what"].notna()].copy()

    panel = (df.pivot_table(
        index=["vintage_q", "target_qp", "scenario"],
        columns="what",
        values="value",
        aggfunc="first"
    ).reset_index())

    # pick "active" scenario per vintage: choose scenario with most non-missing forecasts
    # ----------------------------
    # Choose one scenario per vintage WITHOUT restricting to "PL"
    # ----------------------------

    # Count non-missing forecast entries per vintage & scenario
    counts = (
        panel
        .groupby(["vintage_q", "scenario"])["forecast"]
        .apply(lambda s: s.notna().sum())
        .reset_index(name="n_forecasts")
    )

    # Choose scenario with max forecasts per vintage (tie-break: keep first)
    chosen = counts.loc[counts.groupby("vintage_q")["n_forecasts"].idxmax()].copy()

    # Keep only chosen scenario
    panel_snb = panel.merge(
        chosen[["vintage_q", "scenario"]],
        on=["vintage_q", "scenario"],
        how="inner"
    )

    # horizons
    panel_snb["h_q"] = panel_snb["target_qp"].astype("int64") - panel_snb["vintage_q"].astype("int64")
    panel_snb["h_months"] = panel_snb["h_q"] * 3

    snb_36912 = panel_snb[panel_snb["h_q"].isin([1,2,3,4])].copy()
    snb_36912 = snb_36912.sort_values(["vintage_q","h_q","target_qp"]).reset_index(drop=True)

    snb_path = snb_36912[["vintage_q","h_months","target_qp","scenario","forecast","observed"]].copy()
    return snb_path


# ============================================================
# 2) LOAD YOUR MODEL OUTPUTS (timesafe YoY quantiles)
# ============================================================
def load_model_files(model_files: dict, use_timesafe: bool=True) -> pd.DataFrame:
    out = []
    for h, fp in model_files.items():
        d = pd.read_csv(fp)

        # Parse dates
        d["Date"] = pd.to_datetime(d["Date"])
        d["Target_date"] = pd.to_datetime(d["Target_date"])

        d["h_months"] = h
        d["vintage_q"] = d["Date"].dt.to_period("Q")

        # Choose quantiles
        if use_timesafe:
            q05 = "q05_YoY_timesafe"
            q16 = "q16_YoY_timesafe"
            q84 = "q84_YoY_timesafe"
            q95 = "q95_YoY_timesafe"
        else:
            q05 = "q05_YoY"
            q16 = "q16_YoY"
            q84 = "q84_YoY"
            q95 = "q95_YoY"

        # Some files may not have all columns; fail loudly
        needed = ["Forecast_median_YoY", q05, q16, q84, q95, "Actual_YoY"]
        missing = [c for c in needed if c not in d.columns]
        if missing:
            raise ValueError(f"Missing columns in {fp.name}: {missing}")

        d["median"] = d["Forecast_median_YoY"]
        d["q05"] = d[q05]
        d["q16"] = d[q16]
        d["q84"] = d[q84]
        d["q95"] = d[q95]
        d["realized_yoy"] = d["Actual_YoY"]

        out.append(d[["Date","Target_date","vintage_q","h_months","realized_yoy","median","q05","q16","q84","q95"]].copy())

    model_fc = pd.concat(out, ignore_index=True)
    model_fc = model_fc.sort_values(["vintage_q","h_months"]).reset_index(drop=True)
    return model_fc

#plotting
#------------
def plot_trajectory_grid(df, outpath: Path, title_suffix=""):
    # Create a 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=False, sharey=True)
    horizons = [3, 6, 9, 12]
    axes = axes.flatten()

    for i, h in enumerate(horizons):
        ax = axes[i]
        g = df[df["h_months"] == h].sort_values("vintage_q").copy()
        
        if g.empty:
            ax.set_title(f"No Data for h={h}m")
            continue
            
        x = g["vintage_q"].dt.to_timestamp(how="end")
        
        # Plot SNB and Model Trajectories
        ax.plot(x, g["forecast"], color='tab:red', linewidth=2, label="SNB Point Forecast")
        ax.plot(x, g["median"], color='tab:blue', linewidth=1.5, linestyle='--', label=f"{MODEL_NAME} Median")

        # Plot Probability Bands (68% and 90%)
        ax.fill_between(x, g["q16"], g["q84"], color='tab:blue', alpha=0.3, label="68% Interval")
        ax.fill_between(x, g["q05"], g["q95"], color='tab:blue', alpha=0.1, label="90% Interval")

        # Visual anchors for SNB Price Stability Target (0-2%)
        ax.axhline(2.0, color='black', linewidth=1, linestyle=':', alpha=0.6)
        ax.axhline(0.0, color='black', linewidth=1, linestyle=':', alpha=0.6)

        # Formatting each subplot
        ax.set_title(f"Forecast Horizon: {h} Months", fontsize=14, fontweight='bold')
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        # Minor ticks every year (optional)
        ax.xaxis.set_minor_locator(mdates.YearLocator(1))
        if i >= 2: ax.set_xlabel("Vintage Quarter")
        if i % 2 == 0: ax.set_ylabel("Inflation (YoY %)")
        ax.grid(True, which='both', linestyle='--', alpha=0.5)

    # Single legend for the whole figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=4, fontsize=12)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust to make room for suptitle/legend
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================
# 3) ROUTE A METRICS (descriptive, model-based)
# ============================================================
def inside_interval(x, l, u):
    m = np.isfinite(x) & np.isfinite(l) & np.isfinite(u)
    return np.where(m, (x >= l) & (x <= u), np.nan)

def width(l, u):
    m = np.isfinite(l) & np.isfinite(u)
    w = np.where(m, u - l, np.nan)
    return w

def approx_cdf_piecewise(x, qs, ps):
    """
    Monotone piecewise-linear approximation of CDF using quantile points.
    qs: array of quantile values (must be sortable)
    ps: array of probabilities corresponding to qs
    """
    qs = np.asarray(qs, dtype=float)
    ps = np.asarray(ps, dtype=float)
    ok = np.isfinite(qs) & np.isfinite(ps)
    if ok.sum() < 2:
        return np.nan
    qs, ps = qs[ok], ps[ok]
    order = np.argsort(qs)
    qs, ps = qs[order], ps[order]

    if x <= qs[0]:
        return float(ps[0])
    if x >= qs[-1]:
        return float(ps[-1])
    return float(np.interp(x, qs, ps))


def build_routeA_table(merged: pd.DataFrame) -> pd.DataFrame:
    """
    merged contains: SNB point forecast + your model median/quantiles.
    We DO NOT compute CRPS/RMSE vs SNB. We do:
      - how often SNB point lies inside your intervals (50/68/90)
      - distributional "risk" measures from your model (prob below 0, prob above 2)
      - descriptive deviations between SNB point and model median
    """
    df = merged.copy()

    # --- deviations (trajectory comparison) ---
    df["diff_median_minus_snb"] = df["median"] - df["forecast"]
    df["absdiff_median_minus_snb"] = np.abs(df["diff_median_minus_snb"])

    # --- interval inclusion of SNB point ---
    df["snb_in_68"] = inside_interval(df["forecast"], df["q16"], df["q84"]).astype("float")
    df["snb_in_90"] = inside_interval(df["forecast"], df["q05"], df["q95"]).astype("float")

    df["width_68"] = width(df["q16"], df["q84"])
    df["width_90"] = width(df["q05"], df["q95"])

    # --- model-implied risk probabilities (from piecewise quantile CDF) ---
    # Use (q05,q16,median,q84,q95) at probs (0.05,0.16,0.50,0.84,0.95)
    probs = np.array([0.05, 0.16, 0.50, 0.84, 0.95], dtype=float)

    def p_below(row, thr):
        qs = np.array([row["q05"], row["q16"], row["median"], row["q84"], row["q95"]], dtype=float)
        return approx_cdf_piecewise(thr, qs, probs)

    for thr in TAIL_THRESHOLDS:
        df[f"p_below_{thr:g}"] = df.apply(lambda r: p_below(r, thr), axis=1)

    # Convert above-2
    df["p_above_2"] = 1.0 - df["p_below_2"]

    # Summarize by horizon + overall
    rows = []
    for h, g in df.groupby("h_months"):
        rows.append({
            "h_months": h,
            "n": len(g),
            "mean_abs(median - snb)": np.nanmean(g["absdiff_median_minus_snb"]),
            "mean(median - snb)": np.nanmean(g["diff_median_minus_snb"]),
            "snb_inside_68_rate": np.nanmean(g["snb_in_68"]),
            "snb_inside_90_rate": np.nanmean(g["snb_in_90"]),
            "mean_width_68": np.nanmean(g["width_68"]),
            "mean_width_90": np.nanmean(g["width_90"]),
            "mean_p_below_0": np.nanmean(g["p_below_0"]),
            "mean_p_above_2": np.nanmean(g["p_above_2"]),
        })

    g = df
    rows.append({
        "h_months": "overall",
        "n": len(g),
        "mean_abs(median - snb)": np.nanmean(g["absdiff_median_minus_snb"]),
        "mean(median - snb)": np.nanmean(g["diff_median_minus_snb"]),
        "snb_inside_68_rate": np.nanmean(g["snb_in_68"]),
        "snb_inside_90_rate": np.nanmean(g["snb_in_90"]),
        "mean_width_68": np.nanmean(g["width_68"]),
        "mean_width_90": np.nanmean(g["width_90"]),
        "mean_p_below_0": np.nanmean(g["p_below_0"]),
        "mean_p_above_2": np.nanmean(g["p_above_2"]),
    })

    return df, pd.DataFrame(rows)

# ============================================================
# 4) CONSOLIDATED 2x2 GRID PLOT
# ============================================================
def plot_trajectory_grid(df, outpath: Path, title_suffix=""):
    """
    Creates a 2x2 grid comparing SNB point forecasts to Model Density.
    Aligns with the 'direct' forecasting approach [Lenza et al., 2023].
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), sharey=True)
    horizons = [3, 6, 9, 12]
    axes = axes.flatten()

    for i, h in enumerate(horizons):
        ax = axes[i]
        # Filter for horizon and ensure chronological order
        g = df[df["h_months"] == h].sort_values("vintage_q").copy()
        
        if g.empty:
            ax.text(0.5, 0.5, f"No Data for h={h}m", ha='center')
            continue
            
        # Use string representation of PeriodIndex for X-axis labels
        x = g["vintage_q"].dt.to_timestamp(how="end")

        ax.fill_between(x, g["q05"], g["q95"], color='tab:blue', alpha=0.1, label="90% Interval")
        ax.fill_between(x, g["q16"], g["q84"], color='tab:blue', alpha=0.3, label="68% Interval")

        ax.plot(x, g["median"], color='tab:blue', linestyle='--', linewidth=1.5, label=f"{MODEL_NAME} Median")
        ax.plot(x, g["forecast"], color='tab:red', linewidth=2.5, marker='o', markersize=4, label="SNB Point Forecast")

        import matplotlib.dates as mdates
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_minor_locator(mdates.YearLocator(1))

        # Formatting
        ax.set_title(f"Horizon: {h} Months", fontsize=14, fontweight='bold')

        if i >= 2: ax.set_xlabel("Forecast Vintage (Quarter)")
        if i % 2 == 0: ax.set_ylabel("Inflation (YoY %)")
        ax.grid(True, linestyle='--', alpha=0.3)

    # Global Legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=4, fontsize=12)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    # 1. Load Data
    snb_raw = load_snb_cube(SNB_CSV)
    snb_path = snb_build_path(snb_raw)
    model_fc = load_model_files(MODEL_FILES, use_timesafe=USE_TIMESAFE)

    # 2. Merge - Changed to 'left' to keep recent forecasts even if actuals are missing
    merged = snb_path.merge(model_fc, on=["vintage_q", "h_months"], how="left")
    
    # 3. Clean up - Keep rows where we have both a forecast and a model result
    merged = merged.dropna(subset=["forecast", "median"]).copy()

    # 4. Generate Analytics and Grid Plot
    merged_aug, tableA = build_routeA_table(merged)

    print("\n=== COMPARISON SUMMARY TABLE ===")
    print(tableA.to_string(index=False))

    # Save outputs
    tableA.to_csv(PLOT_DIR / f"comparison_table_{MODEL_NAME}.csv", index=False)
    
    # Plotting
    plot_trajectory_grid(
        merged_aug, 
        PLOT_DIR / f"trajectory_grid_2x2_{MODEL_NAME}.png",
        title_suffix="(timesafe)" if USE_TIMESAFE else ""
    )

    print(f"\nResults saved to: {PLOT_DIR}")