import sys
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import nct
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
    for h in horizons:        
        #loop targets target-wise for current horizon
        for target_name in target_names:  
            #specific column string for target identification
            target_col_str= f"target_{target_name.lower()}_{h}m"            
            #list version for pandas slicing and pca dropping
            target_col_list= [target_col_str]             
            #check existence using string 
            if target_col_str not in df.columns:
                #skip if target column missing from dataset
                print(f"skipping {target_col_str} (not found)")
                continue
            #create system for this horizon including specific target and predictors
            df_system= df[target_col_list + predictor_cols].copy()            
            #get total row count for loop control
            total_rows= len(df_system)
            #get location of evaluation start date in index
            requested_start_idx= df_system.index.get_loc(eval_start_date)
            #handle potential slice return from get_loc
            if isinstance(requested_start_idx, slice):
                requested_start_idx= requested_start_idx.start
            #ensure start index respects training offset
            start_idx= max(requested_start_idx, training_offset)
            #initialize pca holders
            pca_bundle= None
            pca_cols, keep_cols= None, None
            #initialize for recursive predictions
            current_idx= start_idx
            #initialize results list for storage
            results_list= []
            #initialize var to store tuned hyperparams
            tuned_params= None 
            #counter for re-tuning frequency check
            months_since_last_tune= 0
            #set tuning frequency to 3 years
            tune_frequency= 36
            #initialize fixed pca bundle for stability
            pca_bundle_fixed= None
            #pre-fit pca if factors requested
            if use_pca_factors:
                #respect publication lag for factor fit
                pub_lag= 2
                #set end date for pca training sample
                pca_fit_end_date= eval_start_date - pd.DateOffset(months=pub_lag)

                #build clean predictor-only matrix up to fit end date
                df_pca_fit= df_system.loc[:pca_fit_end_date].copy()
                #drop all targets to avoid leakage in factors
                cols_to_drop= [c for c in df_pca_fit.columns if 'target_' in c]
                #remove nans for valid pca fit
                X_pca_fit= df_pca_fit.drop(columns=cols_to_drop).dropna()
                #choose pca vs keep columns once
                pca_cols, keep_cols= get_pca(df_columns=X_pca_fit.columns, target_cols_to_drop=[], target_name=target_name, config=config)

                #fit pca once using time-safe helper
                X_train_factors, X_test_factors, pca_bundle_fixed= make_factor_features_time_safe( X_train=X_pca_fit, X_test=X_pca_fit.tail(1), pca_cols=pca_cols,
                    keep_cols=keep_cols, config=config, forecast_date=pca_fit_end_date,
                    target_name=target_name, h=h, top_k=5, pca_bundle=None)
            #main recursive forecasting loop
            while current_idx <total_rows:
                #get current forecast origin date
                forecast_date= df_system.index[current_idx]
                #calculate realization target date
                target_date= forecast_date+pd.DateOffset(months=h)                
                #skip if not snb policy month
                if forecast_date.month not in snb_months:
                    current_idx+= 1
                    #increment tuning counter monthly
                    months_since_last_tune+= 1
                    continue          

                #skip if target realization not yet available in data
                if target_date not in df_yoy.index:
                    current_idx+= 1
                    continue                
                #respect 2 month publication lag
                pub_lag= 2    
                #calculate end of available training data
                train_end_idx= current_idx - h - pub_lag
                #ensure sufficient data for lags
                if train_end_idx < training_offset + lags:
                    current_idx+= 1
                    continue
                    
                #define indices for slicing training set
                idx_train_end= train_end_idx
                #set observation point for test features
                pub_lag= 2
                t_obs= current_idx -pub_lag
                #ensure test window includes enough history for lags
                test_start= max(training_offset, t_obs -lags)                
                #create initial slices for train and test
                df_train= df_system.iloc[training_offset : idx_train_end + 1].dropna(subset=target_col_list)
                X_test_raw= df_system.iloc[test_start : t_obs + 1].copy()
                
                #identify and separate targets from predictors
                #drop all targets to prevent weights learning target patterns
                cols_to_drop= [c for c in df_train.columns if 'target_' in c]                
                #create pure predictor dataframes
                X_train_predictors= df_train.drop(columns=cols_to_drop)
                X_test_predictors= X_test_raw.drop(columns=cols_to_drop)                
                #save target values for fitting
                y_train= df_train[target_col_list]
                y_test= X_test_raw[target_col_list]
                #transform features into factors if requested
                if use_pca_factors:
                    #identify targets to drop
                    cols_to_drop= [c for c in df_train.columns if 'target_' in c]
                    #isolate predictors
                    X_train_safe= df_train.drop(columns=cols_to_drop)
                    X_test_safe= X_test_raw.drop(columns=cols_to_drop)
                    #apply frozen pca weights to current window
                    X_train_factors, X_test_factors, _= make_factor_features_time_safe(X_train=X_train_safe, X_test=X_test_safe,
                        pca_cols=pca_bundle_fixed["pca_cols"], keep_cols=pca_bundle_fixed["keep_cols"], config=config, forecast_date=forecast_date,
                        target_name=target_name, h=h, top_k=5, pca_bundle=pca_bundle_fixed)
                    #reconstruct training and test matrices with factors
                    df_train= pd.concat([y_train, X_train_factors], axis=1)
                    X_test= pd.concat([y_test, X_test_factors], axis=1)
                    #ensure test columns match train order
                    X_test=X_test.reindex(columns=df_train.columns)
                else:
                    #use raw slices if pca disabled
                    X_test= X_test_raw
                #determine if retuning is required
                should_tune= False
                #tune on first run
                if tuned_params is None:
                    should_tune= True
                #tune every 36 months thereafter
                elif months_since_last_tune >= tune_frequency:
                    should_tune= True 
                    #reset counter after tuning
                    months_since_last_tune= 0                 
                #use previous tuned params or initial defaults
                init_params= tuned_params if tuned_params else initial_prior_params

                #initialize bvar model with specific implementation
                model= BVAR(lags=lags, prior_type=prior_type, prior_params=init_params, implementation_type=implementation_type)
                
                #perform fit with or without hyperparam optimization
                if should_tune:
                    #full fit for hyperparam tuning
                    model.fit(df_train, horizon=h) 
                    #save new tuned params
                    tuned_params= model.params.copy()
                    #reset counter
                    months_since_last_tune= 0   
                else:
                    #fit using fixed lambda from previous tuning
                    if 'independent_niw' in prior_type:
                        #use independent prior fixed lambda
                        model.fit(df_train, horizon=h, fixed_lambda=tuned_params.get('lambda', initial_prior_params.get('lambda', 0.2)))
                    else:
                        #use standard fixed lambda fit
                        model.fit(df_train, horizon=h, fixed_lambda=tuned_params.get('lambda', 0.2))
                    #increment months since tuning
                    months_since_last_tune+= 1
                
                #fix leakage in test set for non-pca path
                if not use_pca_factors:
                    #calculate required rows for lag construction
                    required_rows= h +max([0, 1])+1 
                    #calculate start index for test history
                    start= current_idx-(required_rows- 1)
                    #isolate test window
                    X_test= df_system.iloc[start: current_idx+1]
                    #identify max lag used by model
                    max_lag= max(model.lag_indices) 

                    #overwrite pre-shifted targets with actual known values at origin
                    for tc in target_col_list:
                        col= X_test.columns.get_loc(tc)
                        for L in range(max_lag + 1):
                            #replace future shift with true historical y_t-L
                            X_test= df_system.iloc[start : current_idx+1].copy()
                #generate posterior predictive draws
                preds_draws_all= model.forecast(X_test)                
                #isolate draws for specific target variable
                preds_draws= preds_draws_all[:, 0]
                #extract coefficients for shapley calculation
                x_input_series, coeffs_dict, intercept= model.shapley_params(X_test, 0)

                #calculate shapley values for feature contribution
                shap_dict= shap_values(model_obj=None, X_input=x_input_series, X_train=None, model_type='linear', linear_coeffs=coeffs_dict, linear_const=intercept)

                #direct target evaluation
                #----------------------------
                #setup grid for predictive density evaluation
                eval_quantiles= np.linspace(0.01, 0.99, 99)
                #get actual realized target at forecast origin
                actual_direct= df.loc[forecast_date, target_col_str]
                #skip if target realized value missing
                if pd.isna(actual_direct):
                    current_idx+= 1
                    continue
                #fit skew-t distribution to direct predictive draws
                direct_fit_quantiles= np.percentile(preds_draws, eval_quantiles * 100.0)
                #estimate parametric distribution params
                skew_params_direct= fit_skew_t(direct_fit_quantiles, eval_quantiles)

                #calculate crps scores for performance comparison
                crps_direct_parametric= calculate_crps(actual_direct, skew_params_direct)
                crps_direct_empirical= calculate_crps_quantile(actual_direct, direct_fit_quantiles[None, :], eval_quantiles)
                #handle iterable returns for mean score
                if hasattr(crps_direct_empirical, "__iter__"):
                    crps_direct_empirical= float(np.mean(crps_direct_empirical))

                #calculate pit for calibration check
                pit_direct= stats.nct.cdf(actual_direct, skew_params_direct[0], skew_params_direct[1], loc=skew_params_direct[2], scale=skew_params_direct[3])
                #get distribution parameters for logpdf calculation
                df_nct, nc_nct, loc_nct, scale_nct =skew_params_direct
                logpdf_direct=float(nct.logpdf(actual_direct, df_nct, nc_nct, loc=loc_nct, scale=scale_nct))
                #log predictive density (higher is better).
                logS_direct = logpdf_direct
                #calculate median and point forecast errors
                median_direct= float(np.median(preds_draws))
                rmse_direct= calculate_rmse(actual_direct, median_direct)
                #store direct space shapley values
                final_shap_direct= {f"SHAP_{k}": float(np.asarray(v).squeeze()) for k, v in shap_dict.items()}

                #ex-post yoy evaluation
                #--------------------------------------
                #set target verification date
                T= target_date
                if T not in df_yoy.index:
                    current_idx+= 1
                    continue
                #get realized yoy inflation at target date
                actual_yoy= df_yoy.loc[T, target_name]
                if pd.isna(actual_yoy):
                    current_idx+= 1
                    continue
                #define raw level column and time scaling
                raw_col= f"{target_name}_level"   
                scaling= h /12.0
                #handle 12m horizon specific reconstruction
                if h == 12:
                    #use direct draws for 12m yoy
                    preds_draws_yoy_expost= preds_draws.copy()
                    base_effect_expost= 0.0
                else:
                    #calculate anchor date for 12m yoy
                    lower= T - pd.DateOffset(months=12)
                    if (forecast_date not in df_yoy.index) or (lower not in df_yoy.index):
                        current_idx+= 1
                        continue
                    #calculate log changes for base effect
                    p_t= np.log(df_yoy.loc[forecast_date, raw_col])   
                    p_low= np.log(df_yoy.loc[lower, raw_col])           
                    base_effect_expost= 100.0 *(p_t-p_low)
                    #reconstruct yoy draws from forecast origin
                    preds_draws_yoy_expost= base_effect_expost+ preds_draws*scaling

                #calculate yoy quantiles for plotting
                q05_yoy= float(np.percentile(preds_draws_yoy_expost, 5))
                q16_yoy= float(np.percentile(preds_draws_yoy_expost, 16))
                q84_yoy= float(np.percentile(preds_draws_yoy_expost, 84))
                q95_yoy= float(np.percentile(preds_draws_yoy_expost, 95))
                median_yoy= float(np.median(preds_draws_yoy_expost))

                #time-safe yoy evaluation
                #-----------------------------
                #set publication lag for snb comparison
                pub_lag= 2
                #define date of latest available data at origin
                t_known= forecast_date -pd.DateOffset(months=pub_lag)
                #initialize time-safe scores
                crps_yoy_timesafe_parametric= np.nan
                crps_yoy_timesafe_empirical= np.nan
                if h == 12:
                    #direct draws are time-safe for 12m horizon
                    preds_draws_yoy_timesafe= preds_draws.copy()
                else:
                    #reconstruct using only data available at publication lag
                    lower= T - pd.DateOffset(months=12)
                    if (t_known in df_yoy.index) and (lower in df_yoy.index):
                        #get known log levels
                        p_known= np.log(df_yoy.loc[t_known, raw_col])   
                        p_low= np.log(df_yoy.loc[lower,   raw_col])   
                        #calc base effect from known historical levels
                        base_effect_timesafe= 100.0 * (p_known - p_low)
                        #generate time-safe predictive distribution
                        preds_draws_yoy_timesafe= base_effect_timesafe +preds_draws*scaling
                    else:
                        preds_draws_yoy_timesafe= None
                
                #initialize for quantile evaluation of time-safe yoy distribution
                q05_yoy_timesafe =np.nan
                q16_yoy_timesafe=np.nan
                q84_yoy_timesafe= np.nan
                q95_yoy_timesafe= np.nan
                median_yoy_timesafe=np.nan                
                violation_90_timesafe= np.nan       
                upper_violation_95_timesafe = np.nan  

                #calculate scores if time-safe distribution reconstructed successfully
                if preds_draws_yoy_timesafe is not None:
                    #get quantiles for fit
                    yoy_fit_quantiles= np.percentile(preds_draws_yoy_timesafe, eval_quantiles*100.0)
                    #estimate yoy distribution params
                    skew_params_yoy= fit_skew_t(yoy_fit_quantiles, eval_quantiles)
                    #calc parametric and empirical yoy crps
                    crps_yoy_timesafe_parametric= calculate_crps(actual_yoy, skew_params_yoy)
                    crps_yoy_timesafe_empirical= calculate_crps_quantile(actual_yoy, yoy_fit_quantiles[None, :], eval_quantiles)
                    #get quantiles 
                    q05_yoy_timesafe=float(np.percentile(preds_draws_yoy_timesafe, 5))
                    q16_yoy_timesafe= float(np.percentile(preds_draws_yoy_timesafe, 16))
                    q84_yoy_timesafe= float(np.percentile(preds_draws_yoy_timesafe, 84))
                    q95_yoy_timesafe= float(np.percentile(preds_draws_yoy_timesafe, 95))
                    median_yoy_timesafe=float(np.median(preds_draws_yoy_timesafe))
                    #Bool whether forecast falls outside 90% interval (from 5% and 95% quantiles)
                    violation_90_timesafe=int((actual_yoy <q05_yoy_timesafe) or (actual_yoy >q95_yoy_timesafe))
                    upper_violation_95_timesafe = int(actual_yoy> q95_yoy_timesafe)   #bool if actual > q95

                    #format empirical score
                    if hasattr(crps_yoy_timesafe_empirical, "__iter__"):
                        crps_yoy_timesafe_empirical= float(np.mean(crps_yoy_timesafe_empirical))

                #bundle all metrics into results entry
                results_entry= {'Date': forecast_date, 'Target_date': target_date, 'Actual_direct': float(actual_direct),
                    'Forecast_median_direct': median_direct, 'CRPS_direct_parametric': float(crps_direct_parametric), 'CRPS_direct_empirical': float(crps_direct_empirical), "LogS_direct": logS_direct,
                    'RMSE_direct': float(rmse_direct), 'PIT_direct': float(pit_direct), 'df_skewt_direct': float(skew_params_direct[0]),
                    'nc_skewt_direct': float(skew_params_direct[1]), 'loc_skewt_direct': float(skew_params_direct[2]), 'scale_skewt_direct': float(skew_params_direct[3]),
                    'Actual_YoY': float(actual_yoy), 'Forecast_median_YoY': median_yoy, 'q05_YoY': q05_yoy, 'q16_YoY': q16_yoy, 'q84_YoY': q84_yoy,
                    'q95_YoY': q95_yoy, 'BaseEffect_YoY_expost': float(base_effect_expost), 'CRPS_YoY_timesafe_parametric': float(crps_yoy_timesafe_parametric) if np.isfinite(crps_yoy_timesafe_parametric) else np.nan,
                    'CRPS_YoY_timesafe_empirical': float(crps_yoy_timesafe_empirical) if np.isfinite(crps_yoy_timesafe_empirical) else np.nan, "q05_YoY_timesafe": q05_yoy_timesafe, "q16_YoY_timesafe": q16_yoy_timesafe, "q84_YoY_timesafe": q84_yoy_timesafe, "q95_YoY_timesafe": q95_yoy_timesafe,
                    "Violation90_YoY_timesafe": violation_90_timesafe, "UpperViolation95_YoY_timesafe": upper_violation_95_timesafe}

                #append direct space shap values
                results_entry.update(final_shap_direct)
                #store in results list
                results_list.append(results_entry)
                #move to next window index
                current_idx+= 1

            #save final results to csv for experiment
            results_df= pd.DataFrame(results_list)
            if not results_df.empty:
                #set forecast origin as index
                results_df.set_index('Date', inplace=True)
                #construct unique filename for target and horizon
                save_name= f"Results/Data_experiments_bvar/{config['experiment_name']}_{target_name}_{h}m.csv"
                #export result table
                results_df.to_csv(save_name)

if __name__ =="__main__":
    parser =argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args= parser.parse_args()
    run_experiment(load_config(args.config))
