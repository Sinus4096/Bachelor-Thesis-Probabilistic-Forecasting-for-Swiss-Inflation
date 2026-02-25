import sys
import numpy as np
import pandas as pd
from arch import arch_model
from pathlib import Path
from scipy.stats import nct
import warnings

#get path for utils
current_dir= Path(__file__).resolve().parent
scripts_root= current_dir.parent.parent
sys.path.insert(0, str(scripts_root))

from Scripts.Utils.density_fitting import fit_skew_t
from Scripts.Utils.metrics import (
    calculate_crps,
    calculate_crps_quantile,
    calculate_rmse,
    shap_values,
)

def _get_dist_param_names(dist_obj):
    """
    arch distribution parameter names differ across versions"""
    #check for parameter names attribute in arch object
    names= getattr(dist_obj, "parameter_names", None)
    if names is None:
        return []
    #return list of names regardless of arch version
    return names() if callable(names) else list(names)

def run_experiment():
    #define benchmark experiment name
    experiment_name= "Benchmark_ARGARCH"
    #load stationary feature data
    project_root= current_dir.parent.parent
    data_path= project_root /"Data" /"Cleaned_Data" /"data_stationary.csv"
    df= pd.read_csv(data_path, index_col="Date", parse_dates=True)

    #load yoy data for reconstruction and evaluation
    df_yoy= pd.read_csv(project_root /"Data" /"Cleaned_Data" /"data_yoy.csv", index_col="Date", parse_dates=True)
    #config targets and horizons for benchmark loop
    targets= ["Headline", "Core"]
    retrain_step_months= 3
    horizons= [3, 6, 9, 12]
    eval_start_date= "2012-07-01"
    #only forecast on policy months
    snb_months= [3, 6, 9, 12]

    #dense grid for distribution fitting
    dense_quant= np.linspace(0.01, 0.99, 99)
    #setup timing parameters
    eval_start_dt= pd.Timestamp(eval_start_date)
    #use 7 year rolling window for garch stability
    rolling_window_size= 7 *12
    training_offset= 14
    pub_lag= 2
    #iterate through target variables
    for target_name in targets:
        #iterate through forecast horizons
        for h in horizons:
            #setup column strings based on target choice
            if target_name == "Headline":
                target_col= f"target_headline_{h}m"
                yoy_col= "Headline"
                yoy_raw= "Headline_level"
            else:
                target_col= f"target_core_{h}m"
                yoy_col= "Core"
                yoy_raw= "Core_level"
            #skip if target column not in stationary file
            if target_col not in df.columns:
                continue
            #init holder for recursive results
            recursive_preds= []

            #find start index in time series
            start_idx= df.index.get_loc(eval_start_dt)
            if isinstance(start_idx, slice):
                start_idx= start_idx.start
            total_rows= len(df)
            current_idx= start_idx
            #main recursive window loop
            while current_idx<total_rows:
                #get current origin date
                forecast_date= df.index[current_idx]
                #calc target realization date
                target_date= forecast_date +pd.DateOffset(months=h)
                #only run on specific snb months
                if forecast_date.month not in snb_months:
                    current_idx+= 1
                    continue
                #preshifted direct target series 
                y_full= df[target_col].copy()
                #publication-lag safety in preshifted space
                direct_lag= h +pub_lag
                last_trainable_idx= current_idx-direct_lag
                #skip if not enough data for initial train
                if last_trainable_idx <= training_offset:
                    current_idx+= retrain_step_months
                    continue
                #define training window bounds
                cutoff_date= df.index[last_trainable_idx]
                train_start_date= df.index[training_offset]
                #isolate training target and remove nans
                y_train= y_full.loc[train_start_date:cutoff_date].dropna()
                #apply rolling window if sample exceeds size limit
                if len(y_train) > rolling_window_size:
                    y_train= y_train.iloc[-rolling_window_size:]
                #ensure sufficient observations for garch convergence
                if len(y_train) < 30:
                    current_idx+= retrain_step_months
                    continue

                #fit standard ar(p)-garch(1,1) with skew-t, aic selection
                max_ar= 4
                best_aic= np.inf
                best_res= None
                best_p= 0
                #constant mean baseline fit
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        base_model= arch_model(y_train, mean="Constant", vol="GARCH", p=1, q=1, dist="skewt")
                        res0= base_model.fit(disp="off", show_warning=False)
                    #init best tracker with constant model
                    best_aic, best_res, best_p= res0.aic, res0, 0
                except Exception:
                    pass

                #ar(p) in mean loop to find optimal lag p
                for p_lag in range(1, max_ar +1):
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            model= arch_model(y_train, mean="AR", lags=p_lag, vol="GARCH", p=1, q=1, dist="skewt")
                            res= model.fit(disp="off", show_warning=False)
                        #update if aic improves
                        if res.aic <best_aic:
                            best_aic, best_res, best_p= res.aic, res, p_lag
                    except Exception:
                        continue
                #skip window if fitting failed
                if best_res is None:
                    current_idx+= retrain_step_months
                    continue

                #forecast (iterative)
                #---------------------------------------------------------
                #perform iterative forecast to bridge publication lag
                forecasts= best_res.forecast(horizon=direct_lag, reindex=False)
                #extract point and variance forecasts
                mu_pred= float(forecasts.mean.iloc[-1, -1])
                sigma_pred= float(np.sqrt(forecasts.variance.iloc[-1, -1]))
                #fallback to unconditional std if variance explodes
                if not np.isfinite(sigma_pred) or sigma_pred <= 0:
                    sigma_pred= float(np.std(y_train))
                #distribution params 
                dist_obj= best_res.model.distribution
                dist_param_names= _get_dist_param_names(dist_obj)
                #extract nu and lambda for skew-t distribution
                if not dist_param_names:
                    #fallback for arch versions without name method
                    nu= best_res.params.get("nu", np.nan)
                    lam= best_res.params.get("lambda", np.nan)
                    if not (np.isfinite(nu) and np.isfinite(lam)):
                        current_idx+= retrain_step_months
                        continue
                    dist_params= np.array([float(nu), float(lam)], dtype=float)
                else:
                    try:
                        #standard extraction via named series
                        dist_params= best_res.params.loc[dist_param_names].to_numpy(dtype=float)
                    except Exception:
                        #last-resort parameter retrieval
                        nu= best_res.params.get("nu", np.nan)
                        lam= best_res.params.get("lambda", np.nan)
                        if not (np.isfinite(nu) and np.isfinite(lam)):
                            current_idx+= retrain_step_months
                            continue
                        dist_params= np.array([float(nu), float(lam)], dtype=float)
                #ensure params are finite for distribution ppf
                if np.any(~np.isfinite(dist_params)):
                    current_idx+= retrain_step_months
                    continue

                #shap initialization and calculation for direct target
                shap_dict= {}
                if best_p > 0:
                    #extract most recent lags for shap input
                    X_input= y_train.iloc[-best_p:].copy()
                    #extract intercept from mean equation
                    const_val= float(best_res.params.get("Const", best_res.params.get("mu", 0.0)))
                    #map ar coefficients for shap calc
                    mean_params= {}
                    for i in range(1, best_p+1):
                        key_y= f"y[{i}]"
                        key_ar= f"ar[{i}]"
                        if key_y in best_res.params.index:
                            mean_params[key_y]= float(best_res.params[key_y])
                        elif key_ar in best_res.params.index:
                            mean_params[key_y]= float(best_res.params[key_ar])
                    #calc linear shap values for ar part
                    shap_dict= shap_values(model_obj=None, X_input=X_input, X_train=None,
                        model_type="linear", linear_coeffs=mean_params, linear_const=const_val)
                #format shap dictionary with prefix
                final_shap_direct= {f"SHAP_{k}": float(np.asarray(v).squeeze()) for k, v in shap_dict.items()}

                #direct target evaluation
                #-------------------------------
                #get realized value from stationary file
                actual_direct= df.loc[forecast_date, target_col]
                if pd.isna(actual_direct):
                    current_idx+= retrain_step_months
                    continue
                #generate dense predictive distribution via ppf
                preds_dense_direct= mu_pred +sigma_pred *best_res.model.distribution.ppf(dense_quant, dist_params)
                #smooth empirical distribution into skew-t
                skew_params_direct= fit_skew_t(preds_dense_direct.flatten(), dense_quant)
                #calc direct crps scores
                crps_direct_parametric= float(calculate_crps(actual_direct, skew_params_direct))
                crps_direct_empirical= float(np.mean(calculate_crps_quantile([actual_direct], preds_dense_direct.reshape(1, -1), dense_quant)))

                #calc pit for calibration diagnostics
                pit_direct= float(nct.cdf(actual_direct, skew_params_direct[0], skew_params_direct[1], loc=skew_params_direct[2], scale=skew_params_direct[3]))
                #get distribution parameters for logpdf calculation
                df_nct, nc_nct, loc_nct, scale_nct =skew_params_direct
                logpdf_direct=float(nct.logpdf(actual_direct, df_nct, nc_nct, loc=loc_nct, scale=scale_nct))
                #log predictive density (higher is better).
                logS_direct = logpdf_direct
                #save median and point error
                median_direct= float(mu_pred)
                rmse_direct= float(calculate_rmse(actual_direct, median_direct))

                #ex-post yoy reconstruction 
                #-----------------------------------
                T= target_date
                if T not in df_yoy.index:
                    current_idx+= retrain_step_months
                    continue
                #get yoy realization from data
                actual_yoy= df_yoy.loc[T, yoy_col]
                if pd.isna(actual_yoy):
                    current_idx+= retrain_step_months
                    continue
                #calc reconstruction scaling
                scaling= h/12.0
                if h == 12:
                    #direct target is already yoy for 12m horizon
                    preds_dense_yoy_expost= preds_dense_direct.copy()
                    base_effect_expost= 0.0
                else:
                    #reconstruct yoy from log changes
                    lower= T -pd.DateOffset(months=12)
                    if (forecast_date not in df_yoy.index) or (lower not in df_yoy.index):
                        current_idx+= retrain_step_months
                        continue
                    #get anchor log levels
                    p_t= np.log(df_yoy.loc[forecast_date, yoy_raw])
                    p_low= np.log(df_yoy.loc[lower, yoy_raw])
                    #calc base effect contribution
                    base_effect_expost= 100.0 *(p_t-p_low)
                    #reconstruct yoy dense distribution
                    preds_dense_yoy_expost= base_effect_expost +preds_dense_direct*scaling

                #calc yoy quantiles for fan charts
                q05_yoy= float(np.percentile(preds_dense_yoy_expost, 5))
                q16_yoy= float(np.percentile(preds_dense_yoy_expost, 16))
                q84_yoy= float(np.percentile(preds_dense_yoy_expost, 84))
                q95_yoy= float(np.percentile(preds_dense_yoy_expost, 95))
                median_yoy= float(np.median(preds_dense_yoy_expost))
                #time-safe yoy crps 
                #-------------------
                #get latest known date given publication lag
                t_known= forecast_date-pd.DateOffset(months=pub_lag)
                crps_yoy_timesafe_parametric= np.nan
                crps_yoy_timesafe_empirical= np.nan

                if h == 12:
                    #time-safe equals direct for 12m
                    preds_dense_yoy_timesafe= preds_dense_direct.copy()
                else:
                    #reconstruct only using data known at publication lag
                    lower= T -pd.DateOffset(months=12)
                    if (t_known in df_yoy.index) and (lower in df_yoy.index):
                        p_known= np.log(df_yoy.loc[t_known, yoy_raw])
                        p_low= np.log(df_yoy.loc[lower, yoy_raw])

                        base_effect_timesafe= 100.0 *(p_known-p_low)
                        preds_dense_yoy_timesafe= base_effect_timesafe + preds_dense_direct * scaling
                    else:
                        preds_dense_yoy_timesafe= None
                #initialize for quantile evaluation of time-safe yoy distribution
                q05_yoy_timesafe =np.nan
                q16_yoy_timesafe=np.nan
                q84_yoy_timesafe= np.nan
                q95_yoy_timesafe= np.nan
                median_yoy_timesafe=np.nan                
                violation_90_timesafe= np.nan       
                upper_violation_95_timesafe = np.nan  

                #evaluate time-safe yoy distribution if valid
                if preds_dense_yoy_timesafe is not None:
                    #fit skew-t to yoy quantiles
                    skew_params_yoy= fit_skew_t(preds_dense_yoy_timesafe.flatten(), dense_quant)
                    crps_yoy_timesafe_parametric= float(calculate_crps(actual_yoy, skew_params_yoy))
                    crps_yoy_timesafe_empirical= float(np.mean(calculate_crps_quantile([actual_yoy], preds_dense_yoy_timesafe.reshape(1, -1), dense_quant)))
                    #get quantiles 
                    q05_yoy_timesafe=float(np.percentile(preds_dense_yoy_timesafe, 5))
                    q16_yoy_timesafe= float(np.percentile(preds_dense_yoy_timesafe, 16))
                    q84_yoy_timesafe= float(np.percentile(preds_dense_yoy_timesafe, 84))
                    q95_yoy_timesafe= float(np.percentile(preds_dense_yoy_timesafe, 95))
                    median_yoy_timesafe=float(np.median(preds_dense_yoy_timesafe))
                    #Bool whether forecast falls outside 90% interval (from 5% and 95% quantiles)
                    violation_90_timesafe=int((actual_yoy <q05_yoy_timesafe) or (actual_yoy >q95_yoy_timesafe))
                    upper_violation_95_timesafe = int(actual_yoy> q95_yoy_timesafe)   #bool if actual > q95

                #store results
                result= {"Date": forecast_date, "Target_date": target_date, "Actual_direct": float(actual_direct), "Forecast_median_direct": median_direct,
                    "CRPS_direct_parametric": crps_direct_parametric, "CRPS_direct_empirical": crps_direct_empirical, "LogS_direct": logS_direct, "RMSE_direct": rmse_direct,
                    "PIT_direct": pit_direct, "Actual_YoY": float(actual_yoy), "Forecast_median_YoY": median_yoy, "q05_YoY": q05_yoy,
                    "q16_YoY": q16_yoy, "q84_YoY": q84_yoy, "q95_YoY": q95_yoy, "BaseEffect_YoY_expost": float(base_effect_expost),
                    "CRPS_YoY_timesafe_parametric": float(crps_yoy_timesafe_parametric), "CRPS_YoY_timesafe_empirical": float(crps_yoy_timesafe_empirical), "Forecast_median_YoY_timesafe": median_yoy_timesafe,
                    "q05_YoY_timesafe": q05_yoy_timesafe, "q16_YoY_timesafe": q16_yoy_timesafe, "q84_YoY_timesafe": q84_yoy_timesafe, "q95_YoY_timesafe": q95_yoy_timesafe,
                    "Violation90_YoY_timesafe": violation_90_timesafe, "UpperViolation95_YoY_timesafe": upper_violation_95_timesafe}
                #add shapley contributions to results
                result.update(final_shap_direct)
                recursive_preds.append(result)
                #advance window by retrain step
                current_idx+= retrain_step_months

            #format and save results dataframe
            results_df= pd.DataFrame(recursive_preds)
            #construct output directory and path
            out_dir= project_root /"Results" / "Data_experiments_benchmark"
            out_dir.mkdir(parents=True, exist_ok=True)
            save_path= out_dir /f"{experiment_name}_{target_name}_{h}m.csv"
            #export csv with date index
            if not results_df.empty:
                results_df.set_index("Date", inplace=True)
            results_df.to_csv(save_path)



if __name__ == "__main__":
    #execute benchmark experiment
    run_experiment()