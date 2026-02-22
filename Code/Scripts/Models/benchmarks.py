import sys
import numpy as np
import pandas as pd
from arch import arch_model
from pathlib import Path
from scipy.stats import nct
import warnings
#get path for utils
current_dir=Path(__file__).resolve().parent
#get project root
scripts_root = current_dir.parent.parent   
sys.path.insert(0, str(scripts_root))
from Scripts.Utils.density_fitting import fit_skew_t
from Scripts.Utils.metrics import calculate_crps, calculate_crps_quantile, calculate_rmse, shap_values




def ar_garch_model(y_series, max_ar=4):
    """fits AR(p)-GARCH(1,1) with Student-t errors (p based on AIC)
    """
    #initalize for aic: fit const mean firs to set baseline
    base_model=arch_model(y_series, mean='Constant', vol='GARCH', p=1, q=1, dist='skewt')
    #get results of base model
    best_res= base_model.fit(disp='off', show_warning=False)
    best_aic= best_res.aic #initialize best aic
    #loop throrough AR lag p to find best Mean Equation
    for p in range(1, max_ar + 1):
        #supress warnings during fitting
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")                
            #fit model
            model=arch_model(y_series, mean='AR', lags=p, vol='GARCH', p=1, q=1, dist='skewt')
            #estimate parameters
            res=model.fit(disp='off', show_warning=False)
        #if aic of this model is lower-> update
        if res.aic<best_aic:
            best_aic=res.aic
            best_res =res

            
    #if no AR model worked/increased AIC too much-> simpler const mean model
    if best_res is None:
        model= arch_model(y_series, mean='Constant', vol='GARCH', p=1, q=1, dist='skewt') #-> const mean
        best_res= model.fit(disp='off', show_warning=False)     #estimate parameters
    return best_res

def run_experiment(): 
    #namp experiment for output later 
    experiment_name="Benchmark_ARGARCH"
    #load data
    project_root=current_dir.parent.parent
    #get path to selected data
    data_path =project_root /"Data"/ "Cleaned_Data"/"data_stationary.csv"
    df =pd.read_csv(data_path, index_col='Date', parse_dates=True)
    #load yoy data for evaluation
    df_yoy= pd.read_csv(project_root/"Data"/"Cleaned_Data"/"data_yoy.csv", index_col='Date', parse_dates=True)#load yoy for evaluation
    #need to define stuff, normally defined in the config files
    targets=["Headline", "Core"]  
    retrain_step_months= 3       #re-estimate model every quarter 
    horizons= [3, 6, 9, 12]  #define all horizons
    eval_start_date= "2012-07-01" #start out of sample eval
    #snb forecasts once per quarter: in march, june, september, december
    snb_months=[3, 6, 9, 12]
    #define quantiles (for plotting vs crps calc)
    plot_qunat=[0.05, 0.16, 0.50, 0.84, 0.95]
    dense_quant= np.linspace(0.01, 0.99, 99)
    #get start date as timestamp
    eval_start_dt = pd.Timestamp(eval_start_date)

    #use rolling window to capture structural breaks (set to minimum bc only 25 y data and dont want post covid era to depend on financial cirsis and peg era)
    rolling_window_size=7*12 
    #to match bvar start training date:
    training_offset =14
    #loop through targets and horizons to do recursive forecasts
    for target_name in targets:
        for h in horizons:
            #select cols for target
            if target_name=="Headline":
                target_col= f"target_headline_{h}m"      #which horizon are forecasting
                yoy_col= "Headline" #for evaluation (need to deannualize)
                yoy_raw= "Headline_level"   #want to evaluate on yoy changes
            else:   #same if core
                target_col=f"target_core_{h}m"
                yoy_col= "Core"
                yoy_raw="Core_level"
            #check if target exists in data
            if target_col not in df.columns:
                continue
            #initilalize storage for results
            recursive_preds=[]            
            #numerical index in the dataframe where evaluation starts
            start_idx=df.index.get_loc(eval_start_dt)
            if isinstance(start_idx, slice): # in case of multiple matches take first
                start_idx=start_idx.start
            #get total rows
            total_rows=len(df)
            current_idx =start_idx  #initialize current index for recursive loop

            #recursive forecasting
            # ... inside your recursive_preds loop ...
            
            #recursive forecasting
            while current_idx<total_rows:
                #identify dates now ->when forecast happens
                forecast_date=df.index[current_idx]
                #target date: date we're predictin
                target_date= forecast_date+ pd.DateOffset(months=h)
                
                #check whether is an SNB forecast month
                if forecast_date.month not in snb_months:
                    current_idx+=1   #move to next month
                    continue
                
                #data gets publised 2 months later
                pub_lag = 2
                
                # We need to bridge the gap from Available Data (t-2) to Target (t+h)
                # The total lag in the direct regression is h + pub_lag
                direct_lag = h + pub_lag

                # Calculate the last index where the target is actually known (t-2)
                # In your shifted dataframe, row i contains y_{i+h}.
                # We need y_{i+h} to be <= y_{t-2}.
                # Therefore i <= t - h - 2.
                last_trainable_idx = current_idx - h - pub_lag

                # Check if we have enough data to start
                if last_trainable_idx <= training_offset:
                    current_idx += retrain_step_months
                    continue

                # y is the pre-shifted direct target series (already aligned at origin dates)
                y = df[target_col].copy()

                X = y.shift(direct_lag)

                temp = pd.DataFrame({"y": y, "X": X}).dropna()
                
                # Slice the TRAINING data (strictly stopping at known targets)
                # We use the integer location. The last trainable row is 'last_trainable_idx'.
                
                cutoff_date = df.index[last_trainable_idx]
                train_data = temp.loc[:cutoff_date]

                # Apply Rolling Window
                if len(train_data) > rolling_window_size:
                    train_data = train_data.iloc[-rolling_window_size:]

                # Check minimum sample size
                if len(train_data) < 30:
                    current_idx += retrain_step_months
                    continue

                # Define y_train and X_train for the model
                y_train = train_data['y']
                X_train = train_data[['X']] # Keep as DataFrame for arch

                # --- 2. FIT MODEL (Direct AR-GARCH) ---
                # We use mean='LS' (Least Squares) to allow the custom lagged X
                model = arch_model(y_train, x=X_train, mean='LS', vol='GARCH', p=1, q=1, dist='skewt')
                model_res = model.fit(disp='off', show_warning=False)

                if model_res is None:
                    current_idx += retrain_step_months
                    continue

                # --- 3. FORECAST ---
                # We need the X value for the CURRENT prediction.
                # This is the value at (current_idx - direct_lag) in the original series
                # which corresponds to the last row of 'temp_data' before we sliced it for training.
                
                # Check if we have the observation (t-2)
                obs_idx = current_idx - direct_lag
                if obs_idx < 0:
                    current_idx += retrain_step_months
                    continue
                    
                latest_known_value = df[target_col].iloc[obs_idx]
                
                # Forecast 1 step ahead (conceptually) using the known X
                forecasts = model_res.forecast(horizon=1, x=np.array([[latest_known_value]]), reindex=False)
                
                mu_pred = float(forecasts.mean.iloc[-1, 0])

                sigma_pred = float(np.sqrt(forecasts.variance.iloc[-1, 0]))
                if not np.isfinite(sigma_pred) or sigma_pred <= 0:
                    sigma_pred = float(np.std(y_train))


                #extract skew-t parameters from GARCH fit
                dist_params= model_res.params.iloc[-2:]

                # =========================================================
                # (A) DIRECT TARGET EVALUATION (MAIN COMPARISON)
                # =========================================================

                actual_direct = df.loc[forecast_date, target_col]
                if pd.isna(actual_direct):
                    current_idx += retrain_step_months
                    continue

                # Build predictive distribution in DIRECT space
                # (use skew-t from arch directly)
                preds_dense_direct = mu_pred + sigma_pred * model_res.model.distribution.ppf(dense_quant, dist_params)

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

                # SHAP stays direct (NO YoY scaling)
                const_val = model_res.params.get('Const', 0)
                exclude_params = ['Const', 'omega', 'alpha[1]', 'beta[1]', 'nu', 'lambda']
                mean_params = {k: v for k, v in model_res.params.items()
                            if k not in exclude_params and 'alpha' not in k and 'beta' not in k}

                if not mean_params:
                    mean_params = {k: v for k, v in model_res.params.items() if 'X' in k or 'beta[' in k}

                shap_dict = shap_values(
                    model_obj=None,
                    X_input=X_train,
                    X_train=None,
                    model_type='linear',
                    linear_coeffs=mean_params,
                    linear_const=const_val
                )

                final_shap_direct = {f"SHAP_{k}": float(np.asarray(v).squeeze()) for k, v in shap_dict.items()}

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

                pub_lag = 2
                t_known = forecast_date - pd.DateOffset(months=pub_lag)

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
                #to next window
                current_idx+=retrain_step_months

            #save results
            results_df = pd.DataFrame(recursive_preds)
            if not results_df.empty:
                results_df.set_index('Date', inplace=True)
                save_name = f"Results/Data_experiments_benchmark/{experiment_name}_{target_name}_{h}m.csv"
                results_df.to_csv(save_name)

if __name__ == "__main__":
    run_experiment()

