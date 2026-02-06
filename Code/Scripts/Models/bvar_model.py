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
from Scripts.Utils.metrics import calculate_crps, calculate_crps_quantile, calculate_rmse
from Scripts.Utils.density_fitting import fit_skew_t


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
    training_offset=13
    #set recursive (out-of-sampe) prediction windos (->when stop training and update after how many months)
    eval_start_date= pd.Timestamp(config['data']['eval_start_date'])
    
    #model config from the config file
    model_conf= config['model']
    lags=model_conf.get('lags',2)
    prior_type=model_conf['prior_type']
    implementation_type= model_conf.get('implementation_type','dummies')
    prior_params=model_conf['params']
    
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
        requested_start_idx = df_system.index.get_loc(eval_start_date) #get index of start date

        if isinstance(requested_start_idx, slice):
            requested_start_idx= requested_start_idx.start  #get integer index if slice
        start_idx = max(requested_start_idx, training_offset)
        #initialize for recursive predictions
        current_idx= start_idx
        #initialize dictionary where keys are target names and values are empty lists
        results_storage={t: [] for t in target_names}

        #recursive forecasting loop (do with direct forecasting)
        while current_idx< total_rows:
            #get date
            forecast_date= df_system.index[current_idx]
            target_date=forecast_date+pd.DateOffset(months=h)
            
            #skip if not a forecast month
            if target_date not in df_yoy.index:
                current_idx+= 1
                continue
            #prepare training data up to current idx 
            df_train = df_system.iloc[training_offset : current_idx + 1].dropna(subset=available_targets)
                
            #initialize and fit BVAR model
            model= BVAR(lags=lags, prior_type=prior_type, prior_params=prior_params, implementation_type=implementation_type)
            model.fit(df_train)
            #def test set and include enough previous obs for lags
            X_test = df_system.iloc[current_idx - (training_offset - 1) : current_idx + 1]
            #forecast
            preds_draws_all=model.forecast(X_test)
            #restribt evalluation to quarterly
            if forecast_date.month in snb_months:
                #iterate through all target variables& extract results
                for i, var_name in enumerate(target_names):
                    #preds draws made corresnponding to current target col
                    preds_draws=preds_draws_all[:, i]
                        
                    #get actual yoy value
                    actual_yoy=df_yoy.loc[target_date, var_name]
                    #to calc price levels need levelsof core and headline cpi
                    raw_col= f"{var_name}_level"
                    #direct forecast evaluation: if h==12 can evaluate directly
                    if h==12:
                        preds_draws_yoy=preds_draws
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
                    results_storage[var_name].append({'Date': forecast_date,  'Target_date': target_date,'Actual': actual_yoy,
                                                    'Forecast_median': median, 'q05': np.percentile(preds_draws_yoy, 5), 'q16': np.percentile(preds_draws_yoy, 16),
                                                    'q84': np.percentile(preds_draws_yoy, 84), 'q95': np.percentile(preds_draws_yoy, 95), 'RMSE': rmse,'Empirical_CRPS': empirical_crps,
                                                    'Parametric_CRPS': parametric_crps, 'df_skewt':skew_params[0],'nc_skewt': skew_params[1], 'loc_skewt': skew_params[2],
                                                    'scale_skewt': skew_params[3],'PIT': pit_value})
            #advance window by 1 month
            current_idx +=1

                    
        #save and evaluate final recursive results
        for var_name in target_names:
            results_df = pd.DataFrame(results_storage[var_name])
            results_df.set_index('Date', inplace=True)
                
            save_name = f"Results/Data_experiments_bvar/{config['experiment_name']}_{var_name}_{h}m.csv"
            results_df.to_csv(save_name)

if __name__ =="__main__":
    parser =argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args= parser.parse_args()
    run_experiment(load_config(args.config))
