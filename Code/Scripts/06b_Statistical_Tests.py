#some statistical tests for model outputs
from __future__ import annotations

import os
import re
import glob
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

#plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns

#config
#-----------------------------
#folders containing model experiment results
folders= ["Results/Data_experiments_benchmark", "Results/Data_experiments_bvar", "Results/Data_experiments_qrf"]

#main loss metric for giacomini-white test
PREFERRED_GW_LOSS= "NegLogS_direct"
#fallback loss if logs unavailable
FALLBACK_GW_LOSS= "CRPS_direct_parametric"
#column for coverage violations
VIOL_COL= "Violation90_YoY_timesafe"
#significance level for tests
ALPHA= 0.10
#column for pit values
PIT_COL= "PIT_direct"
#lags for ljung-box test on pit
PIT_LB_LAGS= 12
#lags for newey-west covariance estimation
GW_NW_LAGS= 4

#outputs
#path to save final statistical results
save_csv_path= "Scripts/Plots_and_Tables/06b_Statistical_tests/06b_Statistical_tests.csv"
#directory for significance heatmaps
plots_dir= "Scripts/Plots_and_Tables/06b_Statistical_tests/Plots"

#helpers
#-----------------------------
def parse_filename_info(filename: str) -> Tuple[str, str, Optional[int]]:
    #remove extension from filename
    base= os.path.splitext(filename)[0]
    #split into metadata parts
    parts= base.split('_')
    
    #check if last part is monthly horizon
    if not re.match(r"^\d+m$", parts[-1], re.IGNORECASE):
        return "Unknown", "Unknown", None
    
    try:
        #extract horizon integer
        horizon= int(parts[-1][:-1])
    except ValueError:
        return "Unknown", "Unknown", None
    #initialize target vars
    target= "Unknown"
    target_index= -1
    #search backwards for target keywords
    for i in range(len(parts) - 2, -1, -1):
        p_lower= parts[i].lower()
        if p_lower == "core":
            target= "Core"
            target_index= i
            break
        elif p_lower == "headline":
            target= "Headline"
            target_index= i
            break            
    #check if target identified
    if target == "Unknown":
        return "Unknown", "Unknown", None

    #join parts before target to get model name
    model_name= "_".join(parts[:target_index])
    return model_name, target, horizon

def newey_west_variance_of_mean(x: np.ndarray, L: int) -> float:
    #convert to numeric array
    x= np.asarray(x, float)
    #remove non-finite values
    x= x[np.isfinite(x)]
    #get valid sample size
    T= x.size
    #return nan if insufficient data
    if T <= 1: return np.nan
    #demean series
    x= x - x.mean()
    #calculate variance component at zero lag
    gamma0= float(np.dot(x, x) / T)
    var= gamma0
    #determine max lag index
    max_l= min(L, T -1)
    #loop through lags for autocovariance adjustment
    for l in range(1, max_l + 1):
        #calculate bartlett kernel weight
        w= 1.0 - l / (max_l + 1.0)
        #calculate autocovariance at lag l
        gam= float(np.dot(x[l:], x[:-l]) / T)
        #add weighted contribution
        var+= 2.0*w*gam
    #return scaled variance of mean
    return var /T

def gw_cpa_pvalue(loss_model: np.ndarray, loss_bench: np.ndarray, nw_lags: int= 4):
    from scipy.stats import norm as znorm
    #convert inputs to floats
    loss_model= np.asarray(loss_model, float)
    loss_bench= np.asarray(loss_bench, float)
    #mask for valid data points
    mask= np.isfinite(loss_model) & np.isfinite(loss_bench)
    #calculate loss differential series
    d= (loss_model[mask] - loss_bench[mask]).astype(float)
    #get sample size
    n= int(d.size)
    #return empty results if small sample
    if n < 20: return np.nan, np.nan, n
    #estimate long-run variance of difference
    v= newey_west_variance_of_mean(d, L=nw_lags)
    #check for valid variance estimate
    if not np.isfinite(v) or v <= 1e-12: return np.nan, np.nan, n
    #calculate t-statistic
    stat= float(d.mean() / np.sqrt(v))
    #calculate two-sided p-value
    pval= float(2.0 * (1.0 - znorm.cdf(abs(stat))))
    return stat, pval, n

def _ll_binom(p, k, n):
    #clip probability to avoid log of zero
    p= float(np.clip(p, 1e-12, 1 - 1e-12))
    #return binomial log-likelihood
    return k * np.log(p) + (n - k) * np.log(1 - p)

def christoffersen_lr_pvalues(I: np.ndarray, alpha: float= 0.10) -> Dict[str, float]:
    #prepare indicator array
    I= np.asarray(I, float)
    #drop nans and cast to int
    I= I[np.isfinite(I)].astype(int)
    #get sample length
    T= int(I.size)
    #init result dict
    out= {"n_cov": T, "pi_hat": np.nan, "p_LRuc": np.nan, "p_LRind": np.nan, "p_LRcc": np.nan}
    #skip if not enough observations
    if T < 30:
        if T > 0: out["pi_hat"]= float(I.mean())
        return out
    #total number of violations
    x= int(I.sum())
    #empirical violation rate
    pi_hat= x / T
    out["pi_hat"]= float(pi_hat)
    #likelihood ratio for unconditional coverage
    LRuc= -2.0 * (_ll_binom(alpha, x, T) - _ll_binom(pi_hat, x, T))
    #save uc p-value
    out["p_LRuc"]= float(1.0 - chi2.cdf(LRuc, df=1))    
    #calculate transition counts for independence test
    I0, I1= I[:-1], I[1:]
    #count 0 to 0 transitions
    n00= ((I0 == 0) & (I1 == 0)).sum()
    #count 0 to 1 transitions
    n01= ((I0 == 0) & (I1 == 1)).sum()
    #count 1 to 0 transitions
    n10= ((I0 == 1) & (I1 == 0)).sum()
    #count 1 to 1 transitions
    n11= ((I0 == 1) & (I1 == 1)).sum()
    
    #calculate state transition probabilities
    pi01= n01 /(n00 +n01) if (n00 +n01) > 0 else 0.0
    pi11= n11 / (n10+ n11) if (n10+ n11) > 0 else 0.0
    #unconditional rate for transitions
    pi1= (n01 + n11) / (n00 + n01 + n10 + n11)    
    #calculate log-likelihoods for markov and iid cases
    ll_markov= _ll_binom(pi01, n01, n00 + n01) + _ll_binom(pi11, n11, n10 + n11)
    ll_iid= _ll_binom(pi1, n01 + n11, n00 + n01 + n10 + n11)    
    #likelihood ratio for independence
    LRind= -2.0 * (ll_iid - ll_markov)
    out["p_LRind"]= float(1.0 - chi2.cdf(LRind, df=1))
    #calculate joint test for conditional coverage
    LRcc= LRuc + LRind
    out["p_LRcc"]= float(1.0 - chi2.cdf(LRcc, df=2))
    return out



def pit_ljung_box_pvalue(pit: np.ndarray, lags: int= 12) -> Dict[str, float]:
    #prepare pit values for autocorr check
    pit= np.asarray(pit, float)
    #drop non-numeric
    pit= pit[np.isfinite(pit)]
    #clip bounds for normal transform
    pit= np.clip(pit, 1e-6, 1 - 1e-6)
    #get count
    n= int(pit.size)
    #init result dict
    out= {"n_pit": n, "p_PIT_LB": np.nan, "LB_stat": np.nan, "PIT_mean": np.nan}
    #skip if small sample
    if n < 30:
        if n > 0: out["PIT_mean"]= float(pit.mean())
        return out
    out["PIT_mean"]= float(pit.mean())
    #transform pit to normal space
    z= norm.ppf(pit)
    #demean normalized series
    z= z - z.mean()
    #calculate denominator for autocorr
    denom= float(np.dot(z, z))
    #skip if zero variance
    if not np.isfinite(denom) or denom <= 0: return out
    #check lag count
    max_l= min(lags, n - 1)
    if max_l < 1: return out
    #initialize squared correlation sum
    ac2_sum= 0.0
    #loop through lags for q-statistic
    for k in range(1, max_l + 1):
        #dot product at lag k
        num= float(np.dot(z[k:], z[:-k]))
        #sample autocorrelation
        rho= num / denom
        #weighted squared correlation
        ac2_sum+= (rho * rho) / (n - k)
    #calculate ljung-box statistic
    Q= n * (n + 2.0) * ac2_sum
    out["LB_stat"]= float(Q)
    #calculate p-value from chi2 dist
    out["p_PIT_LB"]= float(1.0 - chi2.cdf(Q, df=max_l))
    return out


#-----------------------------
#plotting
#-----------------------------
def generate_plots(df: pd.DataFrame, out_dir: str):
    #check for output directory existence
    if not os.path.exists(out_dir):
        #create directory if missing
        os.makedirs(out_dir)

    #extract unique experiment combinations
    unique_groups= df[['Target', 'Horizon_m']].drop_duplicates()
    
    for _, row in unique_groups.iterrows():
        #set current target and horizon
        t= row['Target']
        h= row['Horizon_m']
        
        #filter results for current group
        subset= df[(df['Target'] == t) & (df['Horizon_m'] == h)].copy()        
        if subset.empty: continue        
        #pivot statistics for comparison matrix
        pivot_stat= subset.pivot(index='Model', columns='Benchmark', values='GW_stat')
        #pivot p-values for significance stars
        pivot_p= subset.pivot(index='Model', columns='Benchmark', values='GW_p')
        
        #init figure for heatmap
        plt.figure(figsize=(10, 8))        
        #setup labels for annotations
        annot_labels= pivot_stat.fillna(0).round(2).astype(str)        
        #iterate through matrix cells for significance marking
        for i in range(len(pivot_stat.index)):
            for j in range(len(pivot_stat.columns)):
                #get current values
                p_val= pivot_p.iloc[i, j]
                val= pivot_stat.iloc[i, j]
                #handle missing comparisons
                if pd.isna(val): 
                    annot_labels.iloc[i, j]= ""
                    continue                
                #format t-stat string
                txt= f"{val:.2f}"
                #add stars based on p-value thresholds
                if p_val < 0.01: txt+= "**"
                elif p_val < 0.05: txt+= "*"
                elif p_val < 0.10: txt+= "."
                #assign label
                annot_labels.iloc[i, j]= txt

        #generate heatmap plot
        sns.heatmap(pivot_stat, annot=annot_labels, fmt="", cmap="RdBu_r", center=0, linewidths=.5, cbar_kws={'label': 'GW t-stat (Neg = Better)'})    
        #set chart metadata
        plt.title(f"GW Test Statistic: {t} {h}m\n(Negative/Blue = Row Model is Better)")
        plt.tight_layout()
        #save as png
        plt.savefig(os.path.join(out_dir, f"GW_Heatmap_{t}_{h}m.png"))
        #clear memory
        plt.close()

    #drop duplicate rows for model performance summary
    unique_models_df= df.drop_duplicates(subset=['Model', 'Target', 'Horizon_m'])
    
    for t in unique_models_df['Target'].unique():
        #filter for specific target inflation
        subset= unique_models_df[unique_models_df['Target'] == t]
        
        #init bar plot for coverage
        plt.figure(figsize=(12, 6))
        #render bars by model and horizon
        sns.barplot(data=subset, x='Model', y='pi_hat', hue='Horizon_m')        
        #add target reference line
        plt.axhline(0.10, color='red', linestyle='--', label='Target (0.10)')
        #format labels
        plt.xticks(rotation=45, ha='right')
        plt.title(f"Coverage Rate (Target: 0.10) - {t}")
        #set axis bounds
        plt.ylim(0, 0.3) 
        plt.tight_layout()
        #export coverage summary
        plt.savefig(os.path.join(out_dir, f"Coverage_Summary_{t}.png"))
        plt.close()

    #construct text report for diagnostic summary
    report_path= os.path.join(out_dir, "Summary_Report.txt")
    with open(report_path, "w") as f:
        #write report header
        f.write("STATISTICAL TEST SUMMARY \n\n")
        
        #log pit independence results
        f.write("1. PIT Independence Test (Target: p > 0.05)\n")
        f.write("   Models that PASSED (residuals are i.i.d):\n")
        #filter models passing ljung-box test
        passed_pit= unique_models_df[unique_models_df['p_PIT_LB'] > 0.05]
        if passed_pit.empty:
            f.write("   None.\n")
        else:
            for _, row in passed_pit.iterrows():
                f.write(f"   - {row['Model']} ({row['Target']} {row['Horizon_m']}m): p={row['p_PIT_LB']:.3f}\n")        
        #log coverage diagnostic results
        f.write("\n2. Coverage Test (Target: p_LRcc > 0.05)\n")
        f.write("   Models with correct conditional coverage:\n")
        #filter models passing christoffersen test
        passed_cov= unique_models_df[unique_models_df['p_LRcc'] > 0.05]
        if passed_cov.empty:
             f.write("   None.\n")
        else:
            for _, row in passed_cov.iterrows():
                f.write(f"   - {row['Model']} ({row['Target']} {row['Horizon_m']}m): p={row['p_LRcc']:.3f} (Actual={row['pi_hat']:.2f})\n")





#run
#-----------------------------
def main() -> None:
    #init mapping for result dataframes
    by_target_horizon: Dict[Tuple[str, int], Dict[str, pd.DataFrame]]= {}    
    #loop through input folders
    for folder in folders:
        #grab all csv results
        file_paths= glob.glob(os.path.join(folder, "*.csv"))
        for path in file_paths:
            #get filename for parsing
            file_name= os.path.basename(path)
            #extract metadata
            model_name, target, horizon= parse_filename_info(file_name)
            #skip if filename metadata invalid
            if target == "Unknown" or horizon is None: continue
            #load experiment dataframe
            df= pd.read_csv(path)
            #store in nested map
            by_target_horizon.setdefault((target, horizon), {})[model_name]= df

    #check for loaded data
    if not by_target_horizon:
        raise RuntimeError("no result files found.")

    #list to collect testing rows
    rows: List[Dict[str, object]]= []    
    #iterate through combinations chronologically
    for (target, horizon), model_map in sorted(by_target_horizon.items(), key=lambda x: (x[0][0], x[0][1])):
        #get models for current target/horizon
        available_models= sorted(model_map.keys())        
        for model_A_key in available_models:
            #extract primary model dataframe
            df_A= model_map[model_A_key]            
            #calculate single model diagnostics
            cov= {}
            if VIOL_COL in df_A.columns:
                #extract violation indicators
                I= pd.to_numeric(df_A[VIOL_COL], errors="coerce").values
                #run coverage tests
                cov= christoffersen_lr_pvalues(I, alpha=ALPHA)

            pit_res= {}
            if PIT_COL in df_A.columns:
                #extract calibration values
                pit= pd.to_numeric(df_A[PIT_COL], errors="coerce").values
                #run independence test
                pit_res= pit_ljung_box_pvalue(pit, lags=PIT_LB_LAGS)
            #run pairwise accuracy comparisons
            for model_B_key in available_models:
                #extract benchmark model dataframe
                df_B= model_map[model_B_key]
                #identify valid loss column
                gw_loss_col= None
                if PREFERRED_GW_LOSS in df_A.columns and PREFERRED_GW_LOSS in df_B.columns:
                    gw_loss_col= PREFERRED_GW_LOSS
                elif FALLBACK_GW_LOSS in df_A.columns and FALLBACK_GW_LOSS in df_B.columns:
                    gw_loss_col= FALLBACK_GW_LOSS
                
                #init placeholders for test results
                gw_stat, gw_p, gw_n, mean_loss_diff= np.nan, np.nan, np.nan, np.nan
                if gw_loss_col:
                    #handle diagonal self-comparison case
                    if model_A_key == model_B_key:
                        mean_loss_diff, gw_n= 0.0, len(df_A)
                    else:
                        #align models by date for pairwise testing
                        if "Target_date" in df_A.columns and "Target_date" in df_B.columns:
                            #isolate merge keys and loss
                            jA= df_A[["Target_date", gw_loss_col]].copy()
                            jB= df_B[["Target_date", gw_loss_col]].copy()
                            #standardize dates
                            jA["Target_date"]= pd.to_datetime(jA["Target_date"], errors="coerce")
                            jB["Target_date"]= pd.to_datetime(jB["Target_date"], errors="coerce")
                            #inner join series
                            merged= jA.merge(jB, on="Target_date", how="inner", suffixes=("_A", "_B"))
                            #extract matched numeric loss
                            loss_A= pd.to_numeric(merged[f"{gw_loss_col}_A"], errors="coerce").values
                            loss_B= pd.to_numeric(merged[f"{gw_loss_col}_B"], errors="coerce").values
                        else:
                            #fallback to row-wise alignment if dates missing
                            loss_A= pd.to_numeric(df_A[gw_loss_col], errors="coerce").values
                            loss_B= pd.to_numeric(df_B[gw_loss_col], errors="coerce").values
                            #truncate to shortest series
                            min_len= min(len(loss_A), len(loss_B))
                            loss_A, loss_B= loss_A[:min_len], loss_B[:min_len]

                        #calculate accuracy test statistics
                        gw_stat, gw_p, n= gw_cpa_pvalue(loss_A, loss_B, nw_lags=GW_NW_LAGS)
                        gw_n= n
                        #mask valid pairs
                        mask= np.isfinite(loss_A) & np.isfinite(loss_B)
                        #calculate mean error difference
                        if mask.any(): mean_loss_diff= float(np.mean(loss_A[mask] - loss_B[mask]))

                #save results row
                rows.append({"Model": model_A_key, "Benchmark": model_B_key, "Target": target, "Horizon_m": horizon,
                    "GW_mean_loss_diff": mean_loss_diff, "GW_stat": gw_stat, "GW_p": gw_p, "GW_n": gw_n,
                    "pi_hat": cov.get("pi_hat", np.nan), "p_LRuc": cov.get("p_LRuc", np.nan), "p_LRind": cov.get("p_LRind", np.nan), "p_LRcc": cov.get("p_LRcc", np.nan),
                    "PIT_mean": pit_res.get("PIT_mean", np.nan), "p_PIT_LB": pit_res.get("p_PIT_LB", np.nan)})

    #format master results table
    tests_df= pd.DataFrame(rows).sort_values(["Target", "Horizon_m", "Model", "Benchmark"])    
    #set save folder
    out_dir_csv= os.path.dirname(save_csv_path)
    if out_dir_csv: os.makedirs(out_dir_csv, exist_ok=True)
    #write results to csv
    tests_df.to_csv(save_csv_path, index=False)
    print(f"Saved CSV to: {save_csv_path}")
    #render diagnostic heatmaps and summaries
    generate_plots(tests_df, plots_dir)

#main execution entry point
if __name__ == "__main__":
    #run tests
    main()