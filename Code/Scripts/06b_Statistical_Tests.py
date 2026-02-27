from __future__ import annotations

import os
import re
import glob
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

# Plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# CONFIG
# -----------------------------
folders = [
    "Results/Data_experiments_benchmark",
    "Results/Data_experiments_bvar",
    "Results/Data_experiments_qrf",
]

PREFERRED_GW_LOSS = "NegLogS_direct"
FALLBACK_GW_LOSS = "CRPS_direct_parametric"
VIOL_COL = "Violation90_YoY_timesafe"
ALPHA = 0.10
PIT_COL = "PIT_direct"
PIT_LB_LAGS = 12
GW_NW_LAGS = 4

# Outputs
save_csv_path = "Scripts/Plots_and_Tables/06b_Statistical_tests/06b_Statistical_tests.csv"
plots_dir = "Scripts/Plots_and_Tables/06b_Statistical_tests/Plots"

# -----------------------------
# HELPERS
# -----------------------------
def parse_filename_info(filename: str) -> Tuple[str, str, Optional[int]]:
    base = os.path.splitext(filename)[0]
    parts = base.split('_')
    
    if not re.match(r"^\d+m$", parts[-1], re.IGNORECASE):
        return "Unknown", "Unknown", None
    
    try:
        horizon = int(parts[-1][:-1])
    except ValueError:
        return "Unknown", "Unknown", None

    target = "Unknown"
    target_index = -1
    for i in range(len(parts) - 2, -1, -1):
        p_lower = parts[i].lower()
        if p_lower == "core":
            target = "Core"
            target_index = i
            break
        elif p_lower == "headline":
            target = "Headline"
            target_index = i
            break
            
    if target == "Unknown":
        return "Unknown", "Unknown", None

    model_name = "_".join(parts[:target_index])
    return model_name, target, horizon

def newey_west_variance_of_mean(x: np.ndarray, L: int) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    T = x.size
    if T <= 1: return np.nan
    x = x - x.mean()
    gamma0 = float(np.dot(x, x) / T)
    var = gamma0
    max_l = min(L, T - 1)
    for l in range(1, max_l + 1):
        w = 1.0 - l / (max_l + 1.0)
        gam = float(np.dot(x[l:], x[:-l]) / T)
        var += 2.0 * w * gam
    return var / T

def gw_cpa_pvalue(loss_model: np.ndarray, loss_bench: np.ndarray, nw_lags: int = 4):
    from scipy.stats import norm as znorm
    loss_model = np.asarray(loss_model, float)
    loss_bench = np.asarray(loss_bench, float)
    mask = np.isfinite(loss_model) & np.isfinite(loss_bench)
    d = (loss_model[mask] - loss_bench[mask]).astype(float)
    n = int(d.size)
    if n < 20: return np.nan, np.nan, n
    v = newey_west_variance_of_mean(d, L=nw_lags)
    if not np.isfinite(v) or v <= 1e-12: return np.nan, np.nan, n
    stat = float(d.mean() / np.sqrt(v))
    pval = float(2.0 * (1.0 - znorm.cdf(abs(stat))))
    return stat, pval, n

def _ll_binom(p, k, n):
    p = float(np.clip(p, 1e-12, 1 - 1e-12))
    return k * np.log(p) + (n - k) * np.log(1 - p)

def christoffersen_lr_pvalues(I: np.ndarray, alpha: float = 0.10) -> Dict[str, float]:
    I = np.asarray(I, float)
    I = I[np.isfinite(I)].astype(int)
    T = int(I.size)
    out = {"n_cov": T, "pi_hat": np.nan, "p_LRuc": np.nan, "p_LRind": np.nan, "p_LRcc": np.nan}
    if T < 30:
        if T > 0: out["pi_hat"] = float(I.mean())
        return out
    x = int(I.sum())
    pi_hat = x / T
    out["pi_hat"] = float(pi_hat)
    LRuc = -2.0 * (_ll_binom(alpha, x, T) - _ll_binom(pi_hat, x, T))
    out["p_LRuc"] = float(1.0 - chi2.cdf(LRuc, df=1))
    
    I0, I1 = I[:-1], I[1:]
    n00 = ((I0 == 0) & (I1 == 0)).sum()
    n01 = ((I0 == 0) & (I1 == 1)).sum()
    n10 = ((I0 == 1) & (I1 == 0)).sum()
    n11 = ((I0 == 1) & (I1 == 1)).sum()
    
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

def pit_ljung_box_pvalue(pit: np.ndarray, lags: int = 12) -> Dict[str, float]:
    pit = np.asarray(pit, float)
    pit = pit[np.isfinite(pit)]
    pit = np.clip(pit, 1e-6, 1 - 1e-6)
    n = int(pit.size)
    out = {"n_pit": n, "p_PIT_LB": np.nan, "LB_stat": np.nan, "PIT_mean": np.nan}
    if n < 30:
        if n > 0: out["PIT_mean"] = float(pit.mean())
        return out
    out["PIT_mean"] = float(pit.mean())
    z = norm.ppf(pit)
    z = z - z.mean()
    denom = float(np.dot(z, z))
    if not np.isfinite(denom) or denom <= 0: return out
    max_l = min(lags, n - 1)
    if max_l < 1: return out
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
# PLOTTING FUNCTIONS
# -----------------------------
def generate_plots(df: pd.DataFrame, out_dir: str):
    """
    Generates Heatmaps for GW stats and Bar charts for Coverage.
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # 1. Heatmaps (GW Statistic) per Target/Horizon
    # We use the t-statistic because it indicates direction (Better/Worse).
    # Negative stat = Model A < Benchmark (Better).
    
    unique_groups = df[['Target', 'Horizon_m']].drop_duplicates()
    
    for _, row in unique_groups.iterrows():
        t = row['Target']
        h = row['Horizon_m']
        
        subset = df[(df['Target'] == t) & (df['Horizon_m'] == h)].copy()
        
        if subset.empty: continue
        
        # Pivot: Index=Model, Columns=Benchmark, Values=GW_stat
        pivot_stat = subset.pivot(index='Model', columns='Benchmark', values='GW_stat')
        pivot_p = subset.pivot(index='Model', columns='Benchmark', values='GW_p')
        
        plt.figure(figsize=(10, 8))
        
        # Create labels with stars for significance
        annot_labels = pivot_stat.fillna(0).round(2).astype(str)
        
        # Add stars to annotation
        for i in range(len(pivot_stat.index)):
            for j in range(len(pivot_stat.columns)):
                p_val = pivot_p.iloc[i, j]
                val = pivot_stat.iloc[i, j]
                if pd.isna(val): 
                    annot_labels.iloc[i, j] = ""
                    continue
                
                txt = f"{val:.2f}"
                if p_val < 0.01: txt += "**"
                elif p_val < 0.05: txt += "*"
                elif p_val < 0.10: txt += "."
                annot_labels.iloc[i, j] = txt

        # Plot Heatmap
        # Blue (Negative) = Better than Benchmark. Red (Positive) = Worse.
        sns.heatmap(pivot_stat, annot=annot_labels, fmt="", cmap="RdBu_r", center=0, 
                    linewidths=.5, cbar_kws={'label': 'GW t-stat (Neg = Better)'})
        
        plt.title(f"GW Test Statistic: {t} {h}m\n(Negative/Blue = Row Model is Better)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"GW_Heatmap_{t}_{h}m.png"))
        plt.close()

    # 2. Coverage Summary Plot
    # Shows how far pi_hat is from 0.10 for each model
    unique_models_df = df.drop_duplicates(subset=['Model', 'Target', 'Horizon_m'])
    
    for t in unique_models_df['Target'].unique():
        subset = unique_models_df[unique_models_df['Target'] == t]
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=subset, x='Model', y='pi_hat', hue='Horizon_m')
        
        plt.axhline(0.10, color='red', linestyle='--', label='Target (0.10)')
        plt.xticks(rotation=45, ha='right')
        plt.title(f"Coverage Rate (Target: 0.10) - {t}")
        plt.ylim(0, 0.3) # Adjust if coverage is wild
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"Coverage_Summary_{t}.png"))
        plt.close()

    # 3. Simple Text Report for PIT
    report_path = os.path.join(out_dir, "Summary_Report.txt")
    with open(report_path, "w") as f:
        f.write("=== STATISTICAL TEST SUMMARY ===\n\n")
        
        f.write("1. PIT Independence Test (Target: p > 0.05)\n")
        f.write("   Models that PASSED (residuals are i.i.d):\n")
        passed_pit = unique_models_df[unique_models_df['p_PIT_LB'] > 0.05]
        if passed_pit.empty:
            f.write("   None.\n")
        else:
            for _, row in passed_pit.iterrows():
                f.write(f"   - {row['Model']} ({row['Target']} {row['Horizon_m']}m): p={row['p_PIT_LB']:.3f}\n")
        
        f.write("\n2. Coverage Test (Target: p_LRcc > 0.05)\n")
        f.write("   Models with correct conditional coverage:\n")
        passed_cov = unique_models_df[unique_models_df['p_LRcc'] > 0.05]
        if passed_cov.empty:
             f.write("   None.\n")
        else:
            for _, row in passed_cov.iterrows():
                f.write(f"   - {row['Model']} ({row['Target']} {row['Horizon_m']}m): p={row['p_LRcc']:.3f} (Actual={row['pi_hat']:.2f})\n")

    print(f"\n[INFO] Plots and Summary Report saved to: {out_dir}")


# -----------------------------
# MAIN
# -----------------------------
def main() -> None:
    by_target_horizon: Dict[Tuple[str, int], Dict[str, pd.DataFrame]] = {}
    print("Parsing files...")
    
    for folder in folders:
        file_paths = glob.glob(os.path.join(folder, "*.csv"))
        for path in file_paths:
            file_name = os.path.basename(path)
            model_name, target, horizon = parse_filename_info(file_name)
            if target == "Unknown" or horizon is None: continue
            df = pd.read_csv(path)
            by_target_horizon.setdefault((target, horizon), {})[model_name] = df

    if not by_target_horizon:
        raise RuntimeError("No result files found.")

    rows: List[Dict[str, object]] = []
    print("Running comparisons...")
    
    for (target, horizon), model_map in sorted(by_target_horizon.items(), key=lambda x: (x[0][0], x[0][1])):
        available_models = sorted(model_map.keys())
        
        for model_A_key in available_models:
            df_A = model_map[model_A_key]
            
            # Single Model Stats
            cov = {}
            if VIOL_COL in df_A.columns:
                I = pd.to_numeric(df_A[VIOL_COL], errors="coerce").values
                cov = christoffersen_lr_pvalues(I, alpha=ALPHA)

            pit_res = {}
            if PIT_COL in df_A.columns:
                pit = pd.to_numeric(df_A[PIT_COL], errors="coerce").values
                pit_res = pit_ljung_box_pvalue(pit, lags=PIT_LB_LAGS)

            # Pairwise Stats
            for model_B_key in available_models:
                df_B = model_map[model_B_key]
                gw_loss_col = None
                if PREFERRED_GW_LOSS in df_A.columns and PREFERRED_GW_LOSS in df_B.columns:
                    gw_loss_col = PREFERRED_GW_LOSS
                elif FALLBACK_GW_LOSS in df_A.columns and FALLBACK_GW_LOSS in df_B.columns:
                    gw_loss_col = FALLBACK_GW_LOSS
                
                gw_stat, gw_p, gw_n, mean_loss_diff = np.nan, np.nan, np.nan, np.nan

                if gw_loss_col:
                    if model_A_key == model_B_key:
                        mean_loss_diff, gw_n = 0.0, len(df_A)
                    else:
                        if "Target_date" in df_A.columns and "Target_date" in df_B.columns:
                            jA = df_A[["Target_date", gw_loss_col]].copy()
                            jB = df_B[["Target_date", gw_loss_col]].copy()
                            jA["Target_date"] = pd.to_datetime(jA["Target_date"], errors="coerce")
                            jB["Target_date"] = pd.to_datetime(jB["Target_date"], errors="coerce")
                            merged = jA.merge(jB, on="Target_date", how="inner", suffixes=("_A", "_B"))
                            loss_A = pd.to_numeric(merged[f"{gw_loss_col}_A"], errors="coerce").values
                            loss_B = pd.to_numeric(merged[f"{gw_loss_col}_B"], errors="coerce").values
                        else:
                            loss_A = pd.to_numeric(df_A[gw_loss_col], errors="coerce").values
                            loss_B = pd.to_numeric(df_B[gw_loss_col], errors="coerce").values
                            min_len = min(len(loss_A), len(loss_B))
                            loss_A, loss_B = loss_A[:min_len], loss_B[:min_len]

                        gw_stat, gw_p, n = gw_cpa_pvalue(loss_A, loss_B, nw_lags=GW_NW_LAGS)
                        gw_n = n
                        mask = np.isfinite(loss_A) & np.isfinite(loss_B)
                        if mask.any(): mean_loss_diff = float(np.mean(loss_A[mask] - loss_B[mask]))

                rows.append({
                    "Model": model_A_key, "Benchmark": model_B_key, "Target": target, "Horizon_m": horizon,
                    "GW_mean_loss_diff": mean_loss_diff, "GW_stat": gw_stat, "GW_p": gw_p, "GW_n": gw_n,
                    "pi_hat": cov.get("pi_hat", np.nan), "p_LRuc": cov.get("p_LRuc", np.nan),
                    "p_LRind": cov.get("p_LRind", np.nan), "p_LRcc": cov.get("p_LRcc", np.nan),
                    "PIT_mean": pit_res.get("PIT_mean", np.nan), "p_PIT_LB": pit_res.get("p_PIT_LB", np.nan)
                })

    tests_df = pd.DataFrame(rows).sort_values(["Target", "Horizon_m", "Model", "Benchmark"])
    
    # Save CSV
    out_dir_csv = os.path.dirname(save_csv_path)
    if out_dir_csv: os.makedirs(out_dir_csv, exist_ok=True)
    tests_df.to_csv(save_csv_path, index=False)
    print(f"Saved CSV to: {save_csv_path}")

    # Generate Plots
    generate_plots(tests_df, plots_dir)

if __name__ == "__main__":
    main()