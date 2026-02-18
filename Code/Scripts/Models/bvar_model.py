import sys
import pandas as pd
import numpy as np
from scipy import stats
import yaml
import argparse
from pathlib import Path
#get path for utils
current_dir=Path(__file__).resolve().parent
#get project root
scripts_root = current_dir.parent.parent   
sys.path.insert(0, str(scripts_root))
from Scripts.Utils.bvar_utils import BVAR
from Scripts.Utils.metrics import calculate_crps, calculate_crps_quantile, calculate_rmse, shap_values
from Scripts.Utils.density_fitting import fit_skew_t
from Scripts.Utils.qrf_utils import get_pca, make_factor_features_time_safe

def load_config(config_path):
    """
    helper fct to load the config files
    """
    #convert path:ensure it works across different OS/environments
    absolute_path = Path(config_path).resolve()
    with open(absolute_path, 'r') as f:
        return yaml.safe_load(f)
    
#make function out of experiment to run the comparison experiment calling te config files:
def run_experiment(config):
    #want to know which one (which prior in use)
    print(f"run {config['experiment_name']}")
    #load data
    data_filename= config['data'].get('data_file', 'data_stationary_bvar.csv')
    project_root=current_dir.parent.parent
    #get path to selected data
    data_path =project_root /"Data"/ "Cleaned_Data"/data_filename
    df =pd.read_csv(data_path, index_col='Date', parse_dates=True)
    data_yoy_path=project_root/ "Data"/ "Cleaned_Data"/"data_yoy.csv"
    df_yoy=pd.read_csv(data_yoy_path, index_col='Date', parse_dates=True)


    #get target variables and forecast horizon from the config file
    target_names =config['data']['targets']
    horizons=config['data']['horizons']
    #Snb alignment: only forecast in these months
    snb_months=[3,6,9,12]
    #identify predictors
    predictor_cols= [col for col in df.columns if col not in target_names and 'target_' not in col] 
    #define 12-month burn-in period
    training_offset=14
    #set recursive (out-of-sampe) prediction windos (->when stop training and update after how many months)
    eval_start_date= pd.Timestamp(config['data']['eval_start_date'])
    
    #model config from the config file
    model_conf= config['model']
    lags=model_conf.get('lags',12)
    prior_type=model_conf['prior_type']
    implementation_type= model_conf.get('implementation_type','dummies')
    initial_prior_params = model_conf.get('params', {})
    #get whether will use pca factors or not
    use_pca_factors = bool(config.get("model", {}).get("use_pca_factors", False))
    
    #iterate to get right target col
# ... inside run_experiment ...

    # 1. Loop Horizons
    for h in horizons:
        
        # 2. Loop Targets (Target-wise!)
        for target_name in target_names:  # e.g., target_name = "Core"
            
            # --- DEFINE VARIABLES ---
            # The specific column string: "target_core_12m"
            target_col_str = f"target_{target_name.lower()}_{h}m"
            
            # The list version (needed for pandas slicing and PCA dropping): ["target_core_12m"]
            target_col_list = [target_col_str] 
            
            # --- FIX 1: Check existence using the STRING ---
            if target_col_str not in df.columns:
                print(f"Skipping {target_col_str} (not found)")
                continue

            # Identify ALL target columns in the entire dataframe to prevent leakage
            # (We want to drop "target_headline" even when predicting "target_core")
            all_target_cols_drop_list = [c for c in df.columns if 'target_' in c]

            # Create system for this horizon
            # We keep all predictors, plus the specific target we are predicting
            df_system = df[target_col_list + predictor_cols].copy()
            
            total_rows = len(df_system)
            requested_start_idx = df_system.index.get_loc(eval_start_date)
            if isinstance(requested_start_idx, slice):
                requested_start_idx = requested_start_idx.start
            start_idx = max(requested_start_idx, training_offset)

            pca_bundle = None
            pca_cols, keep_cols = None, None

            #initialize for recursive predictions
            current_idx= start_idx
            #initialize dictionary where keys are target names and values are empty lists
            results_list = []
            #initialize var to store tuned params
            tuned_params=None 
            #counter when to re-tune (every 3 years)
            months_since_last_tune=0
            tune_frequency = 36
            #recursive forecasting loop (do with direct forecasting)

            while current_idx< total_rows:
                #get date
                forecast_date= df_system.index[current_idx]
                target_date=forecast_date+pd.DateOffset(months=h)
                #skip if not a training and forecasting month 
                if forecast_date.month not in snb_months:
                    current_idx+= 1
                    months_since_last_tune += 1  #want to check monthly
                    continue          

                #skip if target for evaluation not available (e.g. if we are at the end of the data and do not have the target realized yet)
                if target_date not in df_yoy.index:
                    current_idx+= 1
                    continue
                #targets only know if they finished before forecast date
                train_end_idx= current_idx-h
                if train_end_idx< lags: # Need enough data for lags
                    current_idx+= 1
                    continue
                # Define indices for slicing
                idx_train_end = train_end_idx
                # Ensure we don't go negative on start
                idx_test_start = max(0, current_idx - (h + 2)) 
                # Initial Slices
                # ... inside the while current_idx < total_rows loop ...

                # 1. Create Initial Slices
                df_train = df_system.iloc[training_offset : idx_train_end + 1].dropna(subset=target_col_list)
                X_test_raw = df_system.iloc[idx_test_start : current_idx + 1]
                
                # 2. IDENTIFY AND SEPARATE TARGETS FROM PREDICTORS
                # We must drop ALL target columns from the data used for PCA/Factors
                # otherwise the PCA weights will learn to recognize the target variable.
                cols_to_drop = [c for c in df_train.columns if 'target_' in c]
                
                # Create pure predictor dataframes (No Targets!)
                X_train_predictors = df_train.drop(columns=cols_to_drop)
                X_test_predictors = X_test_raw.drop(columns=cols_to_drop)
                
                # Save targets aside to re-attach later
                # Note: We only need the specific target we are forecasting for the model fitting
                y_train = df_train[target_col_list]
                y_test = X_test_raw[target_col_list]

                if use_pca_factors:
                    # Step A: Variable Selection
                    # Pass the PREDICTOR ONLY dataframe to get_pca
                    # Drop ALL columns containing 'target_' from the dataframe itself
                    cols_to_exclude = [c for c in df_train.columns if 'target_' in c]
                    X_train_safe = df_train.drop(columns=cols_to_exclude)
                    X_test_safe  = X_test_raw.drop(columns=cols_to_exclude)

                    # 2. Now call get_pca using the SAFE dataframe columns
                    # You don't even need 'target_cols_to_drop' anymore because they are gone
                    pca_cols, keep_cols = get_pca(
                        df_columns=X_train_safe.columns, # Pass clean columns
                        target_cols_to_drop=[],          # Redundant now, but keep empty list
                        target_name=target_name, 
                        config=config
                    )
                    # Step B: Factor Creation
                    # Pass the PREDICTOR ONLY dataframes
                    X_train_factors, X_test_factors, _ = make_factor_features_time_safe(
                        X_train=X_train_safe, # <--- No targets here
                        X_test=X_test_safe,   # <--- No targets here
                        pca_cols=pca_cols,
                        keep_cols=keep_cols,
                        config=config,
                        forecast_date=forecast_date,
                        target_name=target_name,
                        h=h,
                        top_k=5,
                        pca_bundle=None 
                    )

                    # Step C: Re-attach Targets for BVAR
                    # BVAR needs [Target, Features]
                    df_train = pd.concat([y_train, X_train_factors], axis=1)
                    X_test   = pd.concat([y_test,  X_test_factors], axis=1)
                    
                    # Ensure columns align perfectly
                    X_test = X_test.reindex(columns=df_train.columns)
                else:
                    # If not using PCA, X_test is just the raw slice
                    X_test = X_test_raw
                # --- FIX leakage: overwrite pre-shifted target at origin row (works for PCA and non-PCA) ---
                # after X_test is FINAL (after the non-PCA slice too)


                    
                #determine if need to retune
                should_tune=False
                #if first run-> tune:
                if tuned_params is None:
                    should_tune =True
                #else every 3 years
                elif months_since_last_tune>= tune_frequency:
                    should_tune=True 
                    months_since_last_tune = 0 #reset counter
                #initialize params to tune -> not tune yet can use default
                init_params= tuned_params if tuned_params else initial_prior_params

                #initialize and fit BVAR model
                model= BVAR(lags=lags, prior_type=prior_type, prior_params=init_params, implementation_type=implementation_type)
                #retune every 3 years
                if should_tune:
                    model.fit(df_train, horizon=h) #without fixed lambda
                    #save best params
                    tuned_params = model.params.copy()
                    months_since_last_tune =0   #reset counter
                #else use lambda from last tuning
                else:
                    if 'independent_niw' in prior_type:
                        model.fit(df_train, horizon=h, fixed_lambda=tuned_params.get('lambda', initial_prior_params.get('lambda', 0.2)))
                    else:
                        model.fit(df_train, horizon=h, fixed_lambda=tuned_params.get('lambda', 0.2))
                    months_since_last_tune += 1
                
                #def test set and include enough previous obs for lags
                if not use_pca_factors:
                # how many rows do we need so forecast() never uses negative idx_y?
                    required_rows = h + max([0, 1]) + 1   # = h + 2 because max lag index = 1

                    start = current_idx - (required_rows - 1)
                    X_test = df_system.iloc[start : current_idx + 1]
                max_lag = max(model.lag_indices)  # = 1
                for tc in target_col_list:
                    col = X_test.columns.get_loc(tc)
                    for L in range(max_lag + 1):  # L=0,1
                        # row for (t-L) currently holds y_{t-L+h}; replace with y_{t-L}
                        X_test.iloc[-1 - L, col] = df_system.iloc[current_idx - h - L][tc]

                
                #forecast
                preds_draws_all=model.forecast(X_test)
                
                
                #preds draws made corresnponding to current target col
                preds_draws=preds_draws_all[:, 0]
                #get params for shaply
                x_input_series, coeffs_dict, intercept= model.shapley_params(X_test, 0)

                #calc shapley values
                shap_dict=shap_values(model_obj=None, X_input=x_input_series, X_train=None, model_type='linear', linear_coeffs=coeffs_dict, linear_const=intercept)

                #get actual yoy value
                actual_yoy=df_yoy.loc[target_date, target_name]
                #to calc price levels need levelsof core and headline cpi
                raw_col= f"{target_name}_level"
                #direct forecast evaluation: if h==12 can evaluate directly
                if h==12:
                    preds_draws_yoy=preds_draws
                    #shapley scaling: standard deviation
                    shap_scaling = 1.0
                    base_shap_effect=0 #since we are forecasting the change from t-12 to t, the base effect is 0 (no change) and the shapley values show how much each feature contributes to deviating from this base effect
                else:
                    #for h<12: combine history with deannualized model predictions
                    months_back=12- h#need the change from t-(12-h) to t
                    history_date= forecast_date-pd.DateOffset(months=months_back)

                    if history_date not in df_yoy.index:
                        current_idx+= 1 
                        continue  #skip if not enough history to deannualize
                    #calc log price levels
                    p_t= np.log(df_yoy.loc[forecast_date, raw_col])
                    p_hist= np.log(df_yoy.loc[history_date, raw_col])   
                    #deannualize model preds
                    base_effect= (p_t-p_hist)*100
                    scaling_factor= h/12
                    preds_draws_yoy= base_effect+(preds_draws*scaling_factor)
                    #shapley scaling: scale the shapley values by the same factor to reflect their contribution to the deannualized forecast
                    shap_scaling= scaling_factor
                    base_shap_effect= base_effect  #the base effect is the deannualized change, and the shapley values show how much each feature contributes to deviating from this deannualized change

                #rescale shapley values from standardized units to YoY units                
                final_shap = {f"SHAP_{k}": float(np.asarray(v).squeeze()) * shap_scaling for k, v in shap_dict.items()}
                final_shap["Shap_Base_Effect"] = float(base_shap_effect)

                #calc CRPS
                eval_quantiles= np.linspace(0.01, 0.99, 99)
                #bvar gives draws so we calc quantiles for the fit
                y_fit_quantiles= np.percentile(preds_draws_yoy, eval_quantiles*100)
                #fit skew-t to the draws
                skew_params= fit_skew_t(y_fit_quantiles, eval_quantiles)
                #empirical crps via quantiles
                empirical_crps = calculate_crps_quantile(actual_yoy, y_fit_quantiles[None, :], eval_quantiles)
                if hasattr(empirical_crps, '__iter__'): empirical_crps=np.mean(empirical_crps)   #if has multiple values average them
                #skew-t crps
                parametric_crps= calculate_crps(actual_yoy, skew_params)
                #PIT value
                pit_value= stats.nct.cdf(actual_yoy, skew_params[0], skew_params[1], loc=skew_params[2], scale=skew_params[3])
                #RMSE 
                median= np.median(preds_draws_yoy) #get median forecast
                rmse= calculate_rmse(actual_yoy, median)
                    
                #store result to specific target list
                results_entry=({'Date': forecast_date,  'Target_date': target_date,'Actual': actual_yoy,
                                                    'Forecast_median': median, 'q05': np.percentile(preds_draws_yoy, 5), 'q16': np.percentile(preds_draws_yoy, 16),
                                                    'q84': np.percentile(preds_draws_yoy, 84), 'q95': np.percentile(preds_draws_yoy, 95), 'RMSE': rmse,'Empirical_CRPS': empirical_crps,
                                                    'Parametric_CRPS': parametric_crps, 'df_skewt':skew_params[0],'nc_skewt': skew_params[1], 'loc_skewt': skew_params[2],
                                                    'scale_skewt': skew_params[3],'PIT': pit_value})
                 #add shapley values to results entry
                results_entry.update(final_shap)                
                results_list.append(results_entry)
                #advance window by 1 month
                current_idx +=1

                        
            #save and evaluate final recursive results
            results_df =pd.DataFrame(results_list)
            if not results_df.empty:
                results_df.set_index('Date', inplace=True)
                save_name = f"Results/Data_experiments_bvar2/{config['experiment_name']}_{target_name}_{h}m.csv"
                results_df.to_csv(save_name)

if __name__ =="__main__":
    parser =argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args= parser.parse_args()
    run_experiment(load_config(args.config))
