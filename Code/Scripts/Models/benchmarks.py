import sys
import numpy as np
import pandas as pd
from arch import arch_model
from pathlib import Path
from scipy.stats import nct
import warnings

# get path for utils
current_dir = Path(__file__).resolve().parent
# get project root
scripts_root = current_dir.parent.parent   
sys.path.insert(0, str(scripts_root))

from Scripts.Utils.density_fitting import fit_skew_t
from Scripts.Utils.metrics import calculate_crps, calculate_crps_quantile, calculate_rmse, shap_values

def run_experiment(): 
    # namp experiment for output later 
    experiment_name = "Benchmark_ARGARCH"
    
    # load data
    project_root = current_dir.parent.parent
    data_path = project_root / "Data" / "Cleaned_Data" / "data_stationary.csv"
    df = pd.read_csv(data_path, index_col='Date', parse_dates=True)
    
    # load yoy data for evaluation
    df_yoy = pd.read_csv(project_root / "Data" / "Cleaned_Data" / "data_yoy.csv", index_col='Date', parse_dates=True)
    
    # define stuff, normally defined in the config files
    targets = ["Headline", "Core"]  
    retrain_step_months = 3       # re-estimate model every quarter 
    horizons = [3, 6, 9, 12]      # define all horizons
    eval_start_date = "2012-07-01" # start out of sample eval
    
    # snb forecasts once per quarter: in march, june, september, december
    snb_months = [3, 6, 9, 12]
    
    # define quantiles (for plotting vs crps calc)
    plot_qunat = [0.05, 0.16, 0.50, 0.84, 0.95]
    dense_quant = np.linspace(0.01, 0.99, 99)
    
    # get start date as timestamp
    eval_start_dt = pd.Timestamp(eval_start_date)

    # use rolling window to capture structural breaks 
    rolling_window_size = 7 * 12 
    # to match bvar start training date:
    training_offset = 14
    
    # loop through targets and horizons to do recursive forecasts
    for target_name in targets:
        for h in horizons:
            # select cols for target
            if target_name == "Headline":
                target_col = f"target_headline_{h}m"      
                yoy_col = "Headline" 
                yoy_raw = "Headline_level"   
            else:   
                target_col = f"target_core_{h}m"
                yoy_col = "Core"
                yoy_raw = "Core_level"
                
            # check if target exists in data
            if target_col not in df.columns:
                continue
                
            # initilalize storage for results
            recursive_preds = []            
            
            # numerical index in the dataframe where evaluation starts
            start_idx = df.index.get_loc(eval_start_dt)
            if isinstance(start_idx, slice): 
                start_idx = start_idx.start
                
            total_rows = len(df)
            current_idx = start_idx  
            pub_lag=2
            # recursive forecasting
            while current_idx < total_rows:
                forecast_date = df.index[current_idx]
                target_date = forecast_date + pd.DateOffset(months=h)

                if forecast_date.month not in snb_months:
                    current_idx += 1
                    continue

                # direct target series
                y_full = df[target_col].copy()

                direct_lag = h + pub_lag
                last_trainable_idx = current_idx - direct_lag

                # match QRF convention: training starts at training_offset
                if last_trainable_idx <= training_offset:
                    current_idx += retrain_step_months
                    continue

                cutoff_date = df.index[last_trainable_idx]
                train_start_date = df.index[training_offset]

                # time-safe training sample
                y_train = y_full.loc[train_start_date:cutoff_date].dropna()

                # rolling window
                if len(y_train) > rolling_window_size:
                    y_train = y_train.iloc[-rolling_window_size:]

                if len(y_train) < 30:
                    current_idx += retrain_step_months
                    continue

                # -----------------------------
                # Fit standard AR(p)-GARCH(1,1)
                # -----------------------------
                max_ar = 4
                best_aic = np.inf
                best_res = None
                best_p = 0

                # constant mean baseline
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        base_model = arch_model(
                            y_train, mean="Constant", vol="GARCH", p=1, q=1, dist="skewt"
                        )
                        res0 = base_model.fit(disp="off", show_warning=False)
                    best_aic = res0.aic
                    best_res = res0
                    best_p = 0
                except Exception:
                    pass

                # AR(p) in mean
                for p_lag in range(1, max_ar + 1):
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            model = arch_model(
                                y_train,
                                mean="AR",
                                lags=p_lag,
                                vol="GARCH",
                                p=1,
                                q=1,
                                dist="skewt",
                            )
                            res = model.fit(disp="off", show_warning=False)
                        if res.aic < best_aic:
                            best_aic = res.aic
                            best_res = res
                            best_p = p_lag
                    except Exception:
                        continue

                if best_res is None:
                    current_idx += retrain_step_months
                    continue

                # -----------------------------
                # Forecast to time t
                # -----------------------------
                forecasts = best_res.forecast(horizon=direct_lag, reindex=False)

                mu_pred = float(forecasts.mean.iloc[-1, -1])
                sigma_pred = float(np.sqrt(forecasts.variance.iloc[-1, -1]))
                if not np.isfinite(sigma_pred) or sigma_pred <= 0:
                    sigma_pred = float(np.std(y_train))

                # robust extraction of skew-t params
                nu = best_res.params.get("nu", np.nan)
                lam = best_res.params.get("lambda", np.nan)
                if not (np.isfinite(nu) and np.isfinite(lam)):
                    current_idx += retrain_step_months
                    continue
                dist_params = np.array([float(nu), float(lam)], dtype=float)

                
                # --- 4. SHAP EXTRACTION (using your shap_values for linear models) ---
                shap_dict = {}
                if best_p > 0:
                    # last p observations available at the forecast origin (cutoff_date)
                    X_input = y_train.iloc[-best_p:].copy()

                    # Constant name in arch can be 'Const' or 'mu'
                    const_val = float(best_res.params.get("Const", best_res.params.get("mu", 0.0)))

                    # Collect AR mean coefficients only (map ar[i] -> y[i] for your shap parser)
                    mean_params = {}
                    for i in range(1, best_p + 1):
                        key_y = f"y[{i}]"
                        key_ar = f"ar[{i}]"
                        if key_y in best_res.params.index:
                            mean_params[key_y] = float(best_res.params[key_y])
                        elif key_ar in best_res.params.index:
                            mean_params[key_y] = float(best_res.params[key_ar])

                    shap_dict = shap_values(
                        model_obj=None,
                        X_input=X_input,          # Series of length p
                        X_train=None,
                        model_type="linear",
                        linear_coeffs=mean_params,
                        linear_const=const_val
                    )

                final_shap_direct = {f"SHAP_{k}": float(np.asarray(v).squeeze()) for k, v in shap_dict.items()}
                # =========================================================
                # (A) DIRECT TARGET EVALUATION (MAIN COMPARISON)
                # =========================================================

                actual_direct = df.loc[forecast_date, target_col]
                if pd.isna(actual_direct):
                    current_idx += retrain_step_months
                    continue

                # Build predictive distribution in DIRECT space
                # (use skew-t from arch directly - updated to best_res)
                preds_dense_direct = mu_pred + sigma_pred * best_res.model.distribution.ppf(dense_quant, dist_params)

                # fit skew-t (same pipeline as other models)
                skew_params_direct = fit_skew_t(preds_dense_direct.flatten(), dense_quant)

                crps_direct_parametric = calculate_crps(actual_direct, skew_params_direct)
                crps_direct_empirical = calculate_crps_quantile(
                    [actual_direct],
                    preds_dense_direct.reshape(1, -1),
                    dense_quant
                )

                pit_direct = nct.cdf(
                    actual_direct,
                    skew_params_direct[0],
                    skew_params_direct[1],
                    loc=skew_params_direct[2],
                    scale=skew_params_direct[3],
                )

                median_direct = float(mu_pred)
                rmse_direct = calculate_rmse(actual_direct, median_direct)

                # =========================================================
                # (B) EX-POST YOY RECONSTRUCTION (PLOTS ONLY)
                # =========================================================

                T = target_date
                if T not in df_yoy.index:
                    current_idx += retrain_step_months
                    continue

                actual_yoy = df_yoy.loc[T, yoy_col]
                if pd.isna(actual_yoy):
                    current_idx += retrain_step_months
                    continue

                raw_col = yoy_raw
                scaling = h / 12.0

                if h == 12:
                    preds_dense_yoy_expost = preds_dense_direct.copy()
                    base_effect_expost = 0.0
                else:
                    lower = T - pd.DateOffset(months=12)
                    if (forecast_date not in df_yoy.index) or (lower not in df_yoy.index):
                        current_idx += retrain_step_months
                        continue

                    p_t = np.log(df_yoy.loc[forecast_date, raw_col])
                    p_low = np.log(df_yoy.loc[lower, raw_col])
                    base_effect_expost = 100.0 * (p_t - p_low)

                    preds_dense_yoy_expost = base_effect_expost + preds_dense_direct * scaling

                # plot quantiles
                q05_yoy = float(np.percentile(preds_dense_yoy_expost, 5))
                q16_yoy = float(np.percentile(preds_dense_yoy_expost, 16))
                q84_yoy = float(np.percentile(preds_dense_yoy_expost, 84))
                q95_yoy = float(np.percentile(preds_dense_yoy_expost, 95))
                median_yoy = float(np.median(preds_dense_yoy_expost))

                # =========================================================
                # (C) TIME-SAFE YOY CRPS (SNB-STYLE)
                # =========================================================

                pub_lag_eval = 2
                t_known = forecast_date - pd.DateOffset(months=pub_lag_eval)

                crps_yoy_timesafe_parametric = np.nan
                crps_yoy_timesafe_empirical = np.nan

                if h == 12:
                    preds_dense_yoy_timesafe = preds_dense_direct.copy()
                else:
                    lower = T - pd.DateOffset(months=12)
                    if (t_known in df_yoy.index) and (lower in df_yoy.index):
                        p_known = np.log(df_yoy.loc[t_known, raw_col])
                        p_low = np.log(df_yoy.loc[lower, raw_col])

                        base_effect_timesafe = 100.0 * (p_known - p_low)
                        preds_dense_yoy_timesafe = base_effect_timesafe + preds_dense_direct * scaling
                    else:
                        preds_dense_yoy_timesafe = None

                if preds_dense_yoy_timesafe is not None:
                    skew_params_yoy = fit_skew_t(preds_dense_yoy_timesafe.flatten(), dense_quant)

                    crps_yoy_timesafe_parametric = calculate_crps(actual_yoy, skew_params_yoy)
                    crps_yoy_timesafe_empirical = calculate_crps_quantile(
                        [actual_yoy],
                        preds_dense_yoy_timesafe.reshape(1, -1),
                        dense_quant
                    )

                # =========================================================
                # STORE RESULTS (same structure as other models)
                # =========================================================

                result = {
                    'Date': forecast_date,
                    'Target_date': target_date,

                    # DIRECT (main comparison)
                    'Actual_direct': float(actual_direct),
                    'Forecast_median_direct': median_direct,
                    'CRPS_direct_parametric': float(crps_direct_parametric),
                    'CRPS_direct_empirical': float(np.mean(crps_direct_empirical)),
                    'RMSE_direct': float(rmse_direct),
                    'PIT_direct': float(pit_direct),

                    # YoY plots
                    'Actual_YoY': float(actual_yoy),
                    'Forecast_median_YoY': median_yoy,
                    'q05_YoY': q05_yoy,
                    'q16_YoY': q16_yoy,
                    'q84_YoY': q84_yoy,
                    'q95_YoY': q95_yoy,
                    'BaseEffect_YoY_expost': float(base_effect_expost),

                    # YoY time-safe metric
                    'CRPS_YoY_timesafe_parametric': float(crps_yoy_timesafe_parametric) if np.isfinite(crps_yoy_timesafe_parametric) else np.nan,
                    'CRPS_YoY_timesafe_empirical': float(np.mean(crps_yoy_timesafe_empirical)) if isinstance(crps_yoy_timesafe_empirical, (list, np.ndarray)) else crps_yoy_timesafe_empirical,
                }

                result.update(final_shap_direct)
                recursive_preds.append(result)
                
                # to next window
                current_idx += retrain_step_months

            # save results
            results_df = pd.DataFrame(recursive_preds)
            if not results_df.empty:
                results_df.set_index('Date', inplace=True)
                save_name = f"Results/Data_experiments_benchmark/{experiment_name}_{target_name}_{h}m.csv"
                # Ensure directories exist
                Path(save_name).parent.mkdir(parents=True, exist_ok=True)
                results_df.to_csv(save_name)

if __name__ == "__main__":
    run_experiment()