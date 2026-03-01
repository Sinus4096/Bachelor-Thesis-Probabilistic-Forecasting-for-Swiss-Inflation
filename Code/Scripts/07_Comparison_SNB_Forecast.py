import pandas as pd
import numpy as np
from pathlib import Path
import re
import warnings
import matplotlib.dates as mdates
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

#config
#--------------------
#get path of current script
SCRIPT_DIR= Path(__file__).resolve().parent
#location of raw inflation and snb data
RAW_DIR= SCRIPT_DIR.parent.parent / "Code" / "Data" / "Raw_Data"
#location of qrf model experiment results
RES_DIR= SCRIPT_DIR.parent.parent / "Code" / "Results" / "Data_experiments_qrf"  

#path to snb conditional forecast comparison file
SNB_CSV= RAW_DIR / "SNB_comparison.csv"

#best model files for headline inflation analysis
MODEL_NAME= "QRF_Default_PCA_Headline"
MODEL_FILES= {3: RES_DIR / "QRF_Default_PCA_Headline_3m.csv", 6: RES_DIR / "QRF_Default_PCA_Headline_6m.csv", 9: RES_DIR / "QRF_Default_PCA_Headline_9m.csv",12: RES_DIR / "QRF_Default_PCA_Headline_12m.csv"}

#use timesafe metrics for honest information set comparison
USE_TIMESAFE= True  
#directory for saving comparison plots
PLOT_DIR= SCRIPT_DIR / "Plots_and_Tables"/"07_Comparison_SNB_Forecast"
#ensure plot directory exists
PLOT_DIR.mkdir(parents=True, exist_ok=True)

#coverage levels for interval evaluation
COVERAGES= [0.68, 0.90]  
#deflation and upper stability thresholds for tail risk
TAIL_THRESHOLDS= [0.0, 2.0]  


#load data
#-------------------
def load_snb_cube(csv_path: Path) -> pd.DataFrame:
    #open snb data cube export
    with open(csv_path, "r", encoding="utf-8") as f:
        #read all lines to find header
        lines= f.readlines()

    start= None
    for i, line in enumerate(lines):
        #search for specific snb semicolon header
        if line.strip().startswith('"Date";"D0";"D1";"Value"'):
            start= i
            break
    if start is None:
        raise ValueError('header row not found in snb csv')
    #load csv starting from identified header row
    df= pd.read_csv(csv_path, sep=";", quotechar='"', skiprows=start, dtype=str)
    #rename snb internal columns to readable names
    df= df.rename(columns={"Date": "target_q", "D0": "series", "D1": "kind", "Value": "value"})
    #strip whitespace from categorical columns
    for c in ["target_q", "series", "kind"]:
        df[c]= df[c].astype(str).str.strip()
    #convert forecast values to numeric
    df["value"]= pd.to_numeric(df["value"], errors="coerce")

    #filter for valid quarterly target strings
    df= df[df["target_q"].str.match(r"^\d{4}-Q[1-4]$", na=False)].copy()
    #filter for valid vintage series codes
    df= df[df["series"].str.match(r"^[MJSD]\d{4}", na=False)].copy()
    #filter for forecast vs observed kinds
    df= df[df["kind"].isin(["P", "BI"])].copy()  
    return df


def snb_build_path(df: pd.DataFrame) -> pd.DataFrame:
    #map publication month letters to integers
    month_map= {"M": 3, "J": 6, "S": 9, "D": 12}
    #extract vintage month and year from series code
    m= df["series"].str.extract(r"^(?P<mon>[MJSD])(?P<year>\d{4})(?P<scenario>.*)$")
    df= df.join(m)

    #convert extracted strings to date info
    df["vintage_month"]= df["mon"].map(month_map)
    df["vintage_year"]= pd.to_numeric(df["year"], errors="coerce")

    #create quarterly period index for vintage
    df["vintage_q"]= pd.PeriodIndex(pd.to_datetime(dict(year=df["vintage_year"], month=df["vintage_month"], day=1)), freq="Q")
    #convert target quarter to period
    df["target_qp"]= pd.PeriodIndex(df["target_q"], freq="Q")
    #label forecast vs observed data points
    df["what"]= df["kind"].map({"P": "forecast", "BI": "observed"})
    #remove entries without label
    df= df[df["what"].notna()].copy()
    #pivot into panel with vintage and target dimensions
    panel= (df.pivot_table(index=["vintage_q", "target_qp", "scenario"], columns="what", values="value", aggfunc="first").reset_index())
    #choose scenario with highest data density per vintage
    counts= (panel.groupby(["vintage_q", "scenario"])["forecast"].apply(lambda s: s.notna().sum()).reset_index(name="n_forecasts"))

    #identify best scenario for each vintage
    chosen= counts.loc[counts.groupby("vintage_q")["n_forecasts"].idxmax()].copy()
    #merge back to filter main panel for chosen scenarios
    panel_snb= panel.merge(chosen[["vintage_q", "scenario"]], on=["vintage_q", "scenario"], how="inner")

    #calculate quarterly and monthly horizons
    panel_snb["h_q"]= panel_snb["target_qp"].astype("int64") -panel_snb["vintage_q"].astype("int64")
    panel_snb["h_months"]= panel_snb["h_q"] *3
    #isolate horizons matching model (3, 6, 9, 12 months)
    snb_36912= panel_snb[panel_snb["h_q"].isin([1,2,3,4])].copy()
    #sort by time for path construction
    snb_36912= snb_36912.sort_values(["vintage_q","h_q","target_qp"]).reset_index(drop=True)
    #select relevant comparison columns
    snb_path= snb_36912[["vintage_q","h_months","target_qp","scenario","forecast","observed"]].copy()
    return snb_path


#load output of QRF PCA model
#------------------------------------------
def load_model_files(model_files: dict, use_timesafe: bool=True) -> pd.DataFrame:
    #init list for model dataframes
    out= []
    for h, fp in model_files.items():
        #load specific horizon csv
        d= pd.read_csv(fp)

        #standardize date columns
        d["Date"]= pd.to_datetime(d["Date"])
        d["Target_date"]= pd.to_datetime(d["Target_date"])
        #store horizon in months
        d["h_months"]= h
        #convert forecast date to quarterly period
        d["vintage_q"]= d["Date"].dt.to_period("Q")
        #select column names based on timesafe preference
        if use_timesafe:
            q05= "q05_YoY_timesafe"
            q16= "q16_YoY_timesafe"
            q84= "q84_YoY_timesafe"
            q95= "q95_YoY_timesafe"
        else:
            q05= "q05_YoY"
            q16= "q16_YoY"
            q84= "q84_YoY"
            q95= "q95_YoY"
        #list columns required for comparison
        needed= ["Forecast_median_YoY", q05, q16, q84, q95, "Actual_YoY"]
        #check for missing data in results
        missing= [c for c in needed if c not in d.columns]
        if missing:
            raise ValueError(f"missing columns in {fp.name}: {missing}")
        #standardize column aliases
        d["median"]= d["Forecast_median_YoY"]
        d["q05"]= d[q05]
        d["q16"]= d[q16]
        d["q84"]= d[q84]
        d["q95"]= d[q95]
        d["realized_yoy"]= d["Actual_YoY"]
        #collect cleaned columns for merge
        out.append(d[["Date","Target_date","vintage_q","h_months","realized_yoy","median","q05","q16","q84","q95"]].copy())

    #combine all horizons into single dataframe
    model_fc= pd.concat(out, ignore_index=True)
    #sort by time and horizon
    model_fc= model_fc.sort_values(["vintage_q","h_months"]).reset_index(drop=True)
    return model_fc


#metrics
#---------------
def inside_interval(x, l, u):
    #check if point forecast lies inside density interval
    m= np.isfinite(x) & np.isfinite(l) & np.isfinite(u)
    return np.where(m, (x >= l) & (x <= u), np.nan)

def width(l, u):
    #calc width of uncertainty bands
    m= np.isfinite(l) & np.isfinite(u)
    w= np.where(m, u - l, np.nan)
    return w

def approx_cdf_piecewise(x, qs, ps):
    """monotone piecewise-linear approximation of cdf"""
    #convert inputs to numeric arrays
    qs= np.asarray(qs, dtype=float)
    ps= np.asarray(ps, dtype=float)
    #filter for valid numeric entries
    ok= np.isfinite(qs) & np.isfinite(ps)
    if ok.sum() < 2:
        return np.nan
    #isolate valid data and sort by quantile value
    qs, ps= qs[ok], ps[ok]
    order= np.argsort(qs)
    qs, ps= qs[order], ps[order]

    #handle out of bounds cases for cdf
    if x <= qs[0]:
        return float(ps[0])
    if x >= qs[-1]:
        return float(ps[-1])
    #linear interpolation for intermediate values
    return float(np.interp(x, qs, ps))


def build_routeA_table(merged: pd.DataFrame) -> pd.DataFrame:
    #copy input dataframe for processing
    df= merged.copy()
    #calculate trajectory deviations
    #subtract snb point from model median
    df["diff_median_minus_snb"]= df["median"] - df["forecast"]
    #get absolute difference magnitude
    df["absdiff_median_minus_snb"]= np.abs(df["diff_median_minus_snb"])
    #calculate interval inclusion rates for snb point
    df["snb_in_68"]= inside_interval(df["forecast"], df["q16"], df["q84"]).astype("float")
    df["snb_in_90"]= inside_interval(df["forecast"], df["q05"], df["q95"]).astype("float")

    #calculate uncertainty band widths
    df["width_68"]= width(df["q16"], df["q84"])
    df["width_90"]= width(df["q05"], df["q95"])
    #define probability levels for quantile interpolation
    probs= np.array([0.05, 0.16, 0.50, 0.84, 0.95], dtype=float)
    def p_below(row, thr):
        #collect quantiles from current row
        qs= np.array([row["q05"], row["q16"], row["median"], row["q84"], row["q95"]], dtype=float)
        #approximate cdf at threshold
        return approx_cdf_piecewise(thr, qs, probs)

    for thr in TAIL_THRESHOLDS:
        #apply probability calculation for each risk threshold
        df[f"p_below_{thr:g}"]= df.apply(lambda r: p_below(r, thr), axis=1)
    #calculate probability of exceeding target (2%)
    df["p_above_2"]= 1.0 -df["p_below_2"]
    #summarize metrics by horizon group
    rows= []
    for h, g in df.groupby("h_months"):
        rows.append({"h_months": h,"n": len(g),
            "mean_abs(median - snb)": np.nanmean(g["absdiff_median_minus_snb"]), "mean(median - snb)": np.nanmean(g["diff_median_minus_snb"]),
            "snb_inside_68_rate": np.nanmean(g["snb_in_68"]), "snb_inside_90_rate": np.nanmean(g["snb_in_90"]),"mean_width_68": np.nanmean(g["width_68"]),
            "mean_width_90": np.nanmean(g["width_90"]), "mean_p_below_0": np.nanmean(g["p_below_0"]),"mean_p_above_2": np.nanmean(g["p_above_2"])})
    #calculate overall summary across all horizons
    g= df
    rows.append({"h_months": "overall", "n": len(g),
        "mean_abs(median - snb)": np.nanmean(g["absdiff_median_minus_snb"]), "mean(median - snb)": np.nanmean(g["diff_median_minus_snb"]), "snb_inside_68_rate": np.nanmean(g["snb_in_68"]),
        "snb_inside_90_rate": np.nanmean(g["snb_in_90"]), "mean_width_68": np.nanmean(g["width_68"]),
        "mean_width_90": np.nanmean(g["width_90"]),"mean_p_below_0": np.nanmean(g["p_below_0"]), "mean_p_above_2": np.nanmean(g["p_above_2"])})

    return df, pd.DataFrame(rows)


#grid plot
#----------------
def plot_trajectory_grid(df, outpath: Path, title_suffix=""):
    #init subplots for horizons comparison
    fig, axes= plt.subplots(2, 2, figsize=(16, 11), sharey=True)
    horizons= [3, 6, 9, 12]
    axes= axes.flatten()
    for i, h in enumerate(horizons):
        #select subplot axis
        ax= axes[i]
        #isolate specific horizon and sort chronologically
        g= df[df["h_months"] == h].sort_values("vintage_q").copy()        
        #handle missing data cells
        if g.empty:
            ax.text(0.5, 0.5, f"no data for h={h}m", ha='center')
            continue
            
        #convert quarterly period to timestamp for x-axis
        x= g["vintage_q"].dt.to_timestamp(how="end")
        #plot wide uncertainty bands (90%)
        ax.fill_between(x, g["q05"], g["q95"], color='tab:blue', alpha=0.1, label="90% Interval")
        #plot narrow uncertainty bands (68%)
        ax.fill_between(x, g["q16"], g["q84"], color='tab:blue', alpha=0.3, label="68% Interval")

        #plot model median forecast
        ax.plot(x, g["median"], color='tab:blue', linestyle='--', linewidth=1.5, label=f"{MODEL_NAME} Median")
        #plot snb point forecast trajectories
        ax.plot(x, g["forecast"], color='tab:red', linewidth=2.5, marker='o', markersize=4, label="SNB Point Forecast")
        #format x-axis with 2-year major ticks
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        #set annual minor ticks
        ax.xaxis.set_minor_locator(mdates.YearLocator(1))
        #set labels and formatting
        ax.set_title(f"Horizon: {h} Months", fontsize=14, fontweight='bold')
        if i >= 2: ax.set_xlabel("Forecast Vintage (Quarter)")
        if i % 2 == 0: ax.set_ylabel("Inflation (YoY %)")
        ax.grid(True, linestyle='--', alpha=0.3)

    #extract plot handles for figure legend
    handles, labels= axes[0].get_legend_handles_labels()
    #place legend at top center
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=4, fontsize=12)
    #rotate x-axis labels
    fig.autofmt_xdate()
    #optimize spacing
    plt.tight_layout()
    #export plot
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()


#run comparison
#-------
if __name__ == "__main__":
    #load raw data sources
    snb_raw= load_snb_cube(SNB_CSV)
    snb_path= snb_build_path(snb_raw)
    model_fc= load_model_files(MODEL_FILES, use_timesafe=USE_TIMESAFE)

    #merge dataframes on vintage and horizon
    merged= snb_path.merge(model_fc, on=["vintage_q", "h_months"], how="left")    
    #remove rows with incomplete comparison data
    merged= merged.dropna(subset=["forecast", "median"]).copy()
    #generate analytics tables and visualization grid
    merged_aug, tableA= build_routeA_table(merged)

    #print comparison results to console
    print(tableA.to_string(index=False))
    #export comparison summary to csv
    tableA.to_csv(PLOT_DIR / f"comparison_table_{MODEL_NAME}.csv", index=False)    
    #execute plot generation
    plot_trajectory_grid(merged_aug, PLOT_DIR / f"trajectory_grid_2x2_{MODEL_NAME}.png", title_suffix="(timesafe)" if USE_TIMESAFE else "")
    #log output location
    print(f"\nresults saved to: {PLOT_DIR}")