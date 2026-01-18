import pandas as pd
import numpy as np
import yaml
import argparse
from pathlib import Path

from Utils.metrics import calculate_crps, qrf_crps_scorer
from Utils.bvar_utils import HierarchicalBayesianRegression
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV


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
    targets=config['data']['targets']
    horizons=config['data']['horizons']
    #set recursive (out-of-sampe) prediction windos (->when stop training and update after how many months)
    eval_start_date= pd.Timestamp(config['data']['eval_start_date'])
    step_months=config['data']['retrain_step_months']

    #iterate through all targets
    for target_name in targets:
        #iterate through all horizons defined in script 03
        for h in horizons:
            #setup data for this specific horizon of specific target (eg 3mont CPI forecast)
            if target_name =="Headline":
                target_col= f"target_headline_{h}m" #set target_col as defined in script 03
            else:
                target_col=f"target_core_{h}m"
            #make sure df_stationary contains the forecast horizon
            if target_col not in df.columns:
                continue

            #to follow the recursive testing, we don't drop rows where target_col is NAN but filter data available at that specific point in time:
            target_cols_to_drop= [col for col in df.columns if 'target_' in col]    #don't want target variable in X later
            recursive_preds=[]  #to store predictions 

            #get index of eval start date
            current_idx= df.index.get_loc(eval_start_date)
            #get integer index if slice
            if isinstance(current_idx, slice):
                current_idx= current_idx.start  
            
            #loop until the end of the data
            total_rows= len(df)
            while current_idx< total_rows:
                #define training data up to current idx, ensure not get below last row
                next_step_idx= min(current_idx+step_months, total_rows)
                #create window of data up to current idx
                df_slice= df.iloc[:next_step_idx].copy()
                X_slice= df_slice.drop(columns=target_cols_to_drop)    #X up to current idx without target cols
                Y_slice= df_slice[target_col]                          #y

                #split into train and test
                train_idx =range(0, current_idx)   #train up to current idx
                test_idx= range(current_idx, next_step_idx)  #test from current idx to next step idx
                #get out of loop if no test data available
                if len(test_idx)==0:
                    break

                #define X and Y as train and test and drop NA's only in train set
                X_train=X_slice.iloc[train_idx]
                Y_train = Y_slice.iloc[train_idx]
                Y_train= Y_train.dropna()   #drop NA's in y_train
                X_train= X_train.loc[Y_train.index]

                X_test=X_slice.iloc[test_idx]
                Y_test= Y_slice.iloc[test_idx]

                #initialize and fit model
                model= HierarchicalBayesianRegression(prior_type=config['model']['prior_type'], shrinkage=config['model'].get('shrinkage', 'hierarchical'),
                                                      params=config['model'].get('params'))
                model.fit(X_train, Y_train)
                preds_draws= model.predict(X_test)   #get predictions
                
                #predict dense grid for CRPS and fan charts
                eval_quantiles= np.linspace(0.01, 0.99, 99)
                #iterate through test idx and store results
                for idx, date_idx in enumerate(test_idx):
                    #get actual y
                    actual_y= Y_test.iloc[idx]
                    #make sure actual y is not NA
                    if pd.isna(actual_y):
                        continue

                    #calc metrics
                    median= np.median(preds_draws[idx,:])
                    q05, q16, q84, q95= np.percentile(preds_draws[idx,:], [5,16,84,95])
                    #calc CRPS
                    preds_dense= np.percentile(preds_draws[idx,:], eval_quantiles*100)   #get dense quantiles
                    crps =calculate_crps([actual_y], [preds_dense], eval_quantiles)  #calc CRPS
                    #average if multiple values returned
                    if hasattr(crps, '__iter__'):
                        crps= np.mean(crps)
                    #store result
                    recursive_preds.append({'Date': df.index[date_idx], 'Actual': actual_y, 'Forecast_median': median,
                                            'q05': q05, 'q16': q16, 'q84': q84, 'q95': q95, 'Steps_CRPS': crps})
                #advance window 
                current_idx= next_step_idx
            
            #save and evaluate final recursive results
            results_df= pd.DataFrame(recursive_preds)
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
