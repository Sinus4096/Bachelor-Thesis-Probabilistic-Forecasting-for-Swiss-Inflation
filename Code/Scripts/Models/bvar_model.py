import sys
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
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
        current_target_cols=[f"target_{var.lower()}_{h}m" for var in target_names]
        #check if all target cols exist
        available_targets = [t for t in current_target_cols if t in df.columns]
        if not available_targets:
            continue
        #create system for this horizon
        df_system=df[available_targets + predictor_cols].copy()
        total_rows= len(df_system)  #total rows in data
        requested_start_idx = df_system.index.get_loc(eval_start_date)     #get index of start date

        if isinstance(requested_start_idx, slice):
            requested_start_idx= requested_start_idx.start    #get integer index if slice
        start_idx = max(requested_start_idx, training_offset)
        #initialize for recursive predictions
        current_idx= start_idx
        #initialize dictionary where keys are target names and values are empty lists
        results_storage={target: [] for target in target_names}
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
            #prepare training data up to current idx 
            df_train = df_system.iloc[training_offset: (current_idx-h)+1].dropna(subset=available_targets)
            X_test = df_system.iloc[current_idx - lags : current_idx + 1]
            #initialize for standardization
            #scaler=StandardScaler()
            #fit scaler on training data predictors
            #df_train=pd.DataFrame(scaler.fit_transform(df_train), index=df_train.index, columns=df_train.columns)
            if use_pca_factors:
                # 1. Store the targets separately before they get lost
                targets_to_keep = df_train[available_targets].copy()
                
                # 2. Decide PCA block vs kept columns
                pca_cols, keep_cols = get_pca(df_columns=df_system.columns, 
                                             target_cols_to_drop=current_target_cols, 
                                             target_name=target_names[1], 
                                             config=config)
                
                # 3. Fit PCA (This returns only features + factors)
                df_train, X_test, pca_info = make_factor_features_time_safe(
                    X_train=df_train, X_test=X_test, 
                    pca_cols=pca_cols, keep_cols=keep_cols,
                    config=config, forecast_date=forecast_date, 
                    target_name=target_names[1], h=h, top_k=5
                )

                # 4. RE-ATTACH TARGETS HERE
                df_train = pd.concat([targets_to_keep, df_train], axis=1)
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
                    model.fit(df_train, horizon=h, fixed_lambda=tuned_params.get('lambda', initial_prior_params.get('lambda', 0.08)))
                else:
                    model.fit(df_train, horizon=h, fixed_lambda=tuned_params.get('lambda', 0.2))
                months_since_last_tune += 1


            #skip if target for evaluation not available (e.g. if we are at the end of the data and do not have the target realized yet)
            if target_date not in df_yoy.index:
                current_idx+= 1
                continue
            #targets only know if they finished before forecast date
            train_end_idx= current_idx-h
            if train_end_idx< lags: # Need enough data for lags
                current_idx+= 1
                continue
            
            #def test set and include enough previous obs for lags
            X_test= df_system.iloc[current_idx- lags : current_idx+ 1]
            #transform test set with scaler fitted on training data
            #X_test= pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
            #forecast
            preds_draws_all=model.forecast(X_test)
            
            #iterate through all target variables& extract results
            for i, var_name in enumerate(target_names):
                #preds draws made corresnponding to current target col
                preds_draws=preds_draws_all[:, i]
                #get name of target col
                target_col_name=current_target_cols[i]
                #get params for shaply
                x_input_series, coeffs_dict, intercept= model.shapley_params(X_test, i)

                #calc shapley values
                shap_dict=shap_values(model_obj=None, X_input=x_input_series, X_train=None, model_type='linear', linear_coeffs=coeffs_dict, linear_const=intercept)
                #get scaler stats for shapley values
                target_col_idx= df_system.columns.get_loc(target_col_name)  #get index of target col in system
                #target_mean= scaler.mean_[target_col_idx]  #get mean of target col from scaler
                #target_scale= scaler.scale_[target_col_idx]  #get scale of target col from scaler
                #unscale the draws for evaluation and shapley value rescaling
                #preds_draws= (preds_draws*target_scale)+target_mean
                #get actual yoy value
                actual_yoy=df_yoy.loc[target_date, var_name]
                #to calc price levels need levelsof core and headline cpi
                raw_col= f"{var_name}_level"
                #direct forecast evaluation: if h==12 can evaluate directly
                if h==12:
                    preds_draws_yoy=preds_draws
                    #shapley scaling: standard deviation
                    #shap_scaling = target_scale
                    base_shap_effect=0 #since we are forecasting the change from t-12 to t, the base effect is 0 (no change) and the shapley values show how much each feature contributes to deviating from this base effect
                else:
                    #for h<12: combine history with deannualized model predictions
                    months_back=12- h#need the change from t-(12-h) to t
                    history_date= forecast_date-pd.DateOffset(months=months_back)

                    if history_date not in df_yoy.index:
                        continue  #skip if not enough history to deannualize
                    #calc log price levels
                    p_t= np.log(df_yoy.loc[forecast_date, raw_col])
                    p_hist= np.log(df_yoy.loc[history_date, raw_col])   
                    #deannualize model preds
                    base_effect= (p_t-p_hist)*100
                    scaling_factor= h/12
                    preds_draws_yoy= base_effect+(preds_draws*scaling_factor)
                    #shapley scaling: scale the shapley values by the same factor to reflect their contribution to the deannualized forecast
                    #shap_scaling= target_scale*scaling_factor
                    base_shap_effect= base_effect  #the base effect is the deannualized change, and the shapley values show how much each feature contributes to deviating from this deannualized change

                #rescale shapley values from standardized units to YoY units
                final_shap={}                
                #for k, v in shap_dict.items():
                    #scale back to original units 
                    #final_shap[k] =v*shap_scaling  
                final_shap['Shap_Base_Effect'] = base_shap_effect
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
                results_storage[var_name].append(results_entry)
            #advance window by 1 month
            current_idx +=1

                    
        #save and evaluate final recursive results
        for var_name in target_names:
            results_df = pd.DataFrame(results_storage[var_name])
            results_df.set_index('Date', inplace=True)
                
            save_name = f"Results/Data_experiments_bvar2/{config['experiment_name']}_{var_name}_{h}m.csv"
            results_df.to_csv(save_name)

if __name__ =="__main__":
    parser =argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args= parser.parse_args()
    run_experiment(load_config(args.config))
