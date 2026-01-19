import pandas as pd
import numpy as np
import yaml
import argparse
from pathlib import Path
from Utils.bvar_utils import BVAR
from Utils.metrics import calculate_crps

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
    #load df_stationary
    path='Code/Data/Cleaned_Data/data_stationary.csv'
    df=pd.read_csv(path, index_col='Date', parse_dates=True)

    #get target variables and forecast horizon from the config file
    target_names =config['data']['targets']
    horizons=config['data']['horizons']
    #identify predictors
    predictor_cols= [col for col in df.columns if col not in target_names]
    #create system with targets first then predictors
    df_system= df[target_names + predictor_cols].copy()
    #set recursive (out-of-sampe) prediction windos (->when stop training and update after how many months)
    eval_start_date= pd.Timestamp(config['data']['eval_start_date'])
    start_idx= df_system.index.get_loc(eval_start_date) #get index of start date
    if isinstance(start_idx, slice):
        start_idx= start_idx.start  #get integer index if slice

    #define max horizon to later simulate the entire path in one loop
    max_h=max(horizons)

    #initialize for recursive predictions
    total_rows= len(df_system)  #total rows in data
    current_idx= start_idx
    recursive_results=[]  #to store results

    #recursive forecasting loop
    while current_idx< total_rows:
        #prepare training data up to current idx
        df_train= df_system.iloc[:current_idx].copy()
        #get which priors to use
        model_conf= config['model']
        #initialize and fit model
        model=BVAR(lags=model_conf.get('lags',2), prior_type=model_conf['prior_type'],prior_params=model_conf['params'])
        #fit model
        model.fit(df_train)

        #simulate forecasts up to max horizon
        forecast_path=model.forecast(df_train, horizon=max_h)
        #get actual observations->errors
        actual_window= df_system.iloc[current_idx: current_idx+max_h]
        #iterate through all horizons to store results
        for h in horizons:
            #skip if horizon longer than available data
            if h> len(actual_window):
                continue
            #convert horizon to index
            step_idx= h-1
            #get date
            date=target= actual_window.index[step_idx]

            #iterate through all target variables& extract results
            for var_name in target_names:
                #find index of variable (have many-> better by name than enumerate)
                v_idx= df_system.columns.get_loc(var_name)
                #get actual value
                actual_y= actual_window.iloc[step_idx, v_idx]
                #get forecast draws
                preds_draws= forecast_path[:, step_idx, v_idx]

                #calc metrics
                median= np.median(preds_draws)
                q05, q16, q84, q95= np.percentile(preds_draws, [5,16,84,95])
                #calc CRPS
                eval_quantiles= np.linspace(0.01, 0.99, 99)
                preds_dense= np.percentile(preds_draws, eval_quantiles*100)   #get dense quantiles
                crps =calculate_crps([actual_y], [preds_dense], eval_quantiles)  #calc CRPS
                #average if multiple values returned
                if hasattr(crps, '__iter__'):
                    crps= np.mean(crps)
                #store result
                recursive_results.append({'Date': date, 'Variable': var_name, 'Horizon': h, 'Actual': actual_y,
                                          'Forecast_median': median, 'q05': q05, 'q16': q16,
                                          'q84': q84, 'q95': q95, 'Steps_CRPS': crps})

            
        #save and evaluate final recursive results
        results_df= pd.DataFrame(recursive_results)
        results_df.set_index('Date', inplace=True)
        out_dir = Path("Results/Data_experiments")
        out_dir.mkdir(parents=True, exist_ok=True)
        save_name=f"Results/Data_experiments/recursive_{config['experiment_name']}_{target_name}_{h}m.csv"
        results_df.to_csv(save_name)


if __name__ =="__main__":
    parser =argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args= parser.parse_args()
    run_experiment(load_config(args.config))
