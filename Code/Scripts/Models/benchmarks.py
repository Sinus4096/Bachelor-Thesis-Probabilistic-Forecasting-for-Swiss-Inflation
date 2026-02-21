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

                # --- 1. PREPARE DIRECT FORECASTING DATA ---
                # We create a temporary dataframe to align Y (target) and X (lagged value)
                # We take data up to the current_idx to ensure we can pull the X for the forecast
                temp_data = pd.DataFrame({'y': df[target_col].iloc[:current_idx+1]})
                
                # Create the regressor X: The target column shifted by (h + 2)
                # This ensures we regress y_{t+h} on y_{t-2}
                temp_data['X'] = temp_data['y'].shift(direct_lag)
                
                # Drop NaNs created by the shift
                temp_data = temp_data.dropna()
                
                # Slice the TRAINING data (strictly stopping at known targets)
                # We use the integer location. The last trainable row is 'last_trainable_idx'.
                # We must find the corresponding timestamp or integer location in our temp_data
                train_mask = (df.index[0] + pd.to_timedelta(np.arange(len(df)), unit='D')) # specific index handling not needed if we rely on iloc
                
                # Simpler slicing: define cut-off integer relative to the original df
                # We map the global integer index to the temp_data
                train_data = temp_data.iloc[: (last_trainable_idx - training_offset)] 
                # Note: exact slicing depends on how much data was dropped. 
                # Safer method: Slice by Date Index
                cutoff_date = df.index[last_trainable_idx]
                train_data = temp_data.loc[:cutoff_date]

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

                # variance fallback if degenerate
                if model_res.params.get('omega', 0) < 1e-6 and model_res.params.get('alpha[1]', 0) < 1e-6:
                    sigma_pred = float(np.std(y_train))
                else:
                    sigma_pred = float(np.sqrt(forecasts.variance.iloc[-1, 0]))

                #extract skew-t parameters from GARCH fit
                dist_params= model_res.params.iloc[-2:]

                #reconstruct to yoy changes: check if target date exists in yoy data
                # df_yoy is already shifted to availability time
                # Check target availability
                if target_date not in df_yoy.index:
                    current_idx += retrain_step_months
                    continue

                # Define critical dates strictly based on information available
                T = target_date                     # The future date we are predicting
                t_now = forecast_date               # Today
                t_known = t_now - pd.DateOffset(months=pub_lag) # The last data we actually have
                
                # For YoY calculation: Inflation ~ ln(Price_T) - ln(Price_T-12)
                # We need the base price from T-12
                lower = T - pd.DateOffset(months=12)

                # Check if all necessary dates exist in the index
                if (t_known not in df_yoy.index) or (lower not in df_yoy.index) or (T not in df_yoy.index):
                    current_idx += retrain_step_months
                    continue
                scaling_factor = h / 12

                actual_val = df_yoy.loc[T, yoy_col]

                # Get the Price Levels (Log)
                # CRITICAL FIX: Use t_known (t-2), NOT t0 (t)
                p_known = np.log(df_yoy.loc[t_known, yoy_raw]) 
                p_base  = np.log(df_yoy.loc[lower, yoy_raw])
                
                # Base Effect Calculation
                # This calculates the momentum committed up to the last KNOWN data point.
                # Note: The gap between p_known and p_target is covered by your model's forecast.
                base_effect = 100 * (p_known - p_base)

                # SCALING LOGIC
                # This depends on exactly what 'mu_pred' represents.
                # If mu_pred is the "Average Monthly Inflation Rate" forecasted over the gap:
                # The gap we need to bridge is from t_known to T.
                # Total months = h + pub_lag
                
                total_gap_months = h + pub_lag
                
                # Assuming mu_pred is a monthly rate (or scaled similarly):
                # We project the price forward: P_target = P_known + (Forecast * Gap)
                # Then subtract P_base to get YoY.
                
                # Note: Adjust 'scaling_factor' based on your specific target definition.
                # If mu_pred is already scaled to the horizon h, you might need to adjust.
                # Assuming simple linear projection of the rate:
                
                projected_growth = mu_pred * (total_gap_months / 12) # Or however your target is scaled
                
                # IF your target was already "Inflation over h months":
                # Then you just add mu_pred. 
                # But since you used 'scaling_factor = h/12' before, I assume mu_pred is annualized.
                
                # Corrected logic using your previous scaling style but applied to the full gap:
                # mu_yoy = (Price_Known - Price_Base) + (Forecasted_Change_over_Gap)
                
                # Let's stick to your logic but anchor correctly:
                # You want to estimate P_T. 
                # P_T_est = p_known + (mu_pred_deannualized * total_gap_months)
                
                # If mu_pred is annualized:
                monthly_drift = mu_pred / scaling_factor # Recover monthly drift (approx)
                if scaling_factor == 0: monthly_drift = 0 # safety
                
                # Re-apply drift over the TRUE unknown period (h + 2 months)
                # This makes the forecast responsible for the 2 missing months + h future months
                prediction_gap_component = mu_pred * (total_gap_months / 12) 
                
                # Alternatively, if you want to keep it simple and your mu_pred specifically 
                # covers the shift we did in step 1, just use the base_effect of p_known.
                
                mu_yoy = base_effect + (mu_pred * (total_gap_months/h)) 
                # Note: The math above depends heavily on if 'mu_pred' is a RATE or a LEVEL.
                # Given your previous code, let's try the safest "No-Peek" version:
                
                mu_yoy = base_effect + mu_pred * (total_gap_months/12)
                sigma_yoy = sigma_pred * (total_gap_months/12)

                #add robustness check
                sigma_yoy=max(sigma_yoy, y_train.std() * 0.8)
                #ask model how many params needed
                n_shape_params= model_res.model.distribution.num_params
                #extract correct dist params
                dist_params= model_res.params.iloc[-n_shape_params:]
                #reconstrunct quantiles to yoy
                preds_plot_yoy = mu_yoy + sigma_yoy * model_res.model.distribution.ppf(plot_qunat, dist_params)
                preds_dense_yoy = mu_yoy + sigma_yoy * model_res.model.distribution.ppf(dense_quant, dist_params)

                #shapley values: need params
                const_val=model_res.params.get('Const', 0)  #get constant
                
                # UPDATED SHAP LOGIC for LS Model
                # In LS models, the param is named 'X' (from the column name) or 'beta[0]'
                # We filter for anything that isn't Const, omega, alpha, beta(garch), or shape params
                exclude_params = ['Const', 'omega', 'alpha[1]', 'beta[1]', 'nu', 'lambda']
                mean_params = {k: v for k, v in model_res.params.items() 
                               if k not in exclude_params and 'alpha' not in k and 'beta' not in k}
                # Fallback: explicitly look for 'X' or 'y['
                if not mean_params:
                     mean_params = {k: v for k, v in model_res.params.items() if 'X' in k or 'y[' in k or 'beta[' in k}

                #call fct for shap values - pass X_train (the regressor) as input
                shap_dict=shap_values(model_obj=None, X_input=X_train, X_train=None, model_type='linear', linear_coeffs=mean_params, linear_const=const_val)
                
                #initialize dict to store shap values
                final_shap = {}
                #apply scaling logic to shap results  
                final_shap['Shap_Base_Effect'] =base_effect #add base effect
                #scale output
                for k, v in shap_dict.items():
                    final_shap[k] =v* scaling_factor
                
                #get rmse of meadian forecast
                sq_error =calculate_rmse(actual_val, preds_plot_yoy[2])

                # ... (rest of your code continues from here) ...
                #use skew t to fit data for parametric crps eval
                y_fit_data=preds_dense_yoy.flatten() #data to fit
                skew_params=fit_skew_t(y_fit_data, dense_quant)
                #calc pit values for crps eval
                pit_val=nct.cdf(actual_val, skew_params[0], skew_params[1], loc=skew_params[2], scale=skew_params[3])
                #calc parametric crps based on fitted skew-t
                param_crps= calculate_crps(actual_val, skew_params)
                #empirical crps to see what fitting cost us
                empirical_crps= calculate_crps_quantile([actual_val], preds_dense_yoy.reshape(1,-1), dense_quant)
                
                #store results
                result= {'Date': forecast_date, 'Target_date': target_date, 'Actual': actual_val, 'Forecast_median': preds_plot_yoy[2], 
                    'q05': preds_plot_yoy[0], 'q16': preds_plot_yoy[1], 'q84': preds_plot_yoy[3], 'q95': preds_plot_yoy[4], 'Squared_Error': sq_error,
                    'Empirical_CRPS': empirical_crps, 'Parametric_CRPS': param_crps, 'PIT': pit_val, 'df_skewt': skew_params[0], 'nc_skewt': skew_params[1], 
                    'loc_skewt': skew_params[2], 'scale_skewt': skew_params[3]}
                #add Shapley values to the result dictionary 
                result.update(final_shap)
                #append to recursive preds
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

