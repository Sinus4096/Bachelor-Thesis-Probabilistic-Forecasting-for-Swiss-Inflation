import os
from pathlib import Path
import sys
import pandas as pd
from pyparsing import Path
import numpy as np
from quantile_forest import RandomForestQuantileRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
import yaml
import argparse
from scipy.stats import nct
#get path for utils
current_dir=Path(__file__).resolve().parent
#get project root
project_root= current_dir.parent.parent
sys.path.append(str(project_root))
#import needed utils
from Utils.metrics import qrf_crps_scorer, calculate_crps
from Utils.density_fitting import fit_skew_t

#use config files in order to run once Meinshausens default qrf and once a qrf with hyperparameter tuning
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
    #want to know which one (default or tuning)
    print(f"run {config['experiment_name']}")
    #load data_stationary and data_yoy
    path='Code/Data/Cleaned_Data/data_stationary.csv'
    df=pd.read_csv(path, index_col='Date', parse_dates=True)
    df_yoy=pd.read_csv('Code/Data/Cleaned_Data/data_yoy.csv', index_col='Date', parse_dates=True)

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
                yoy_col="Headline"  #evaluate headline data
                yoy_raw="Headline_level"  #for conditional forecasts
            else:
                target_col=f"target_core_{h}m"
                yoy_col="Core"  #evaluate core data
                yoy_raw="Core_level"  #for conditional forecasts
            #make sure df_stationary contains the forecast horizon
            if target_col not in df.columns:
                continue
            #to follow the recursive testing, we don't drop rows where target_col is NAN but filter data available at that specific point in time:
            target_cols_to_drop= [col for col in df.columns if 'target_' in col]    #don't want target variable in X later

            final_params={} #initialize params for loop
            #check if we need to tune or default
            if config['model'].get('tune_hyperparameters', False):
                #get data to tune on (up until eval_start_date)
                split_idx= df.index.searchsorted(eval_start_date)
                df_tune= df.iloc[:split_idx].copy().dropna(subset=[target_col]) #define df for tuning up until eval_start_date
                #define X and Y 
                X_tune =df_tune.drop(columns=target_cols_to_drop)
                y_tune=df_tune[target_col]
                #get model infos
                raw_grid= config['model']['param_grid']
                #initialize grids
                search_space={}
                fixed_params ={}
                #randomizedsearchcv needs lists-> transfrom
                for k, v in raw_grid.items():
                    if isinstance(v, list):
                        search_space[k] = v
                    else:
                        fixed_params[k] = v
                #define model for cross-validation
                tscv=TimeSeriesSplit(n_splits=5)
                base_model= RandomForestQuantileRegressor(**fixed_params)
                search = RandomizedSearchCV(estimator=base_model, param_distributions=search_space, n_iter=15, scoring=qrf_crps_scorer,
                                            cv=tscv, n_jobs=-1, random_state=42)
                search.fit(X_tune, y_tune)

                #combine fixed parameters with best found params
                final_params= {**fixed_params, **search.best_params_}
            #if don't need to tune: dfault
            else:
                final_params= config['model']['params']
            
            #recursive out-of-sample predictions
            recursive_preds = []    #initialize storage for out-of-sample predictions
            #start time loop at eval_start_date-> get index location of eval_start_date 
            start_idx = df.index.get_loc(eval_start_date)
            if isinstance(start_idx, slice):    #if get_loc returns slice->handle
                start_idx= start_idx.start
            total_rows= len(df) #get length of the original df

            #initialize the recursive loop and then iterate til to end of df
            current_idx= start_idx
            while current_idx <total_rows:
                #define the windows:
                next_step_idx= min(current_idx+step_months, total_rows) #define the next quarter 
                train_indices= range(0, current_idx)    #training set ranges from beginning up to the current index
                test_indices = range(current_idx, next_step_idx)    #testin from current index up until next quarter

                if len (test_indices)==0:
                    break #get out if reached the last row already

                #define ssubdf of original df up until next forecast step
                df_slice=df.iloc[:next_step_idx].copy()

                #separate X and Y
                X_slice= df_slice.drop(columns=target_cols_to_drop) #drop all cols starting with target_
                Y_slice= df_slice[target_col]
                #define train and test set
                X_train= X_slice.iloc[train_indices]
                Y_train = Y_slice.iloc[train_indices]
                X_test = X_slice.iloc[test_indices]
                #don't use df for testing /evaluating but the  yoy changes
                #only drop NAN's for the training set: test might have NAN's in the end-> inference
                Y_train= Y_train.dropna()
                X_train= X_train.loc[Y_train.index]


                #use final params determined by which model use
                model_args=final_params.copy()
                #ensure reproducibility
                model_args['random_state']=42

                #train model
                model= RandomForestQuantileRegressor(**model_args)
                model.fit(X_train, Y_train)
                #predict key quantiles for evaluation and plotting
                plot_quantiles=[0.05, 0.16, 0.50, 0.84, 0.95]    
                preds_plot= model.predict(X_test, quantile=plot_quantiles)    #pre safe the predictions
                
                #predict dense grid for CRPS and fan charts
                eval_quantiles= np.linspace(0.01, 0.99, 99)
                preds_dense = model.predict(X_test, quantile=eval_quantiles)
                #store the resultes nicer in predefined list
                for idx, date_idx in enumerate(test_indices):
                     #get date/origin of forecast
                     forecast_date= df.index[date_idx]
                     #calc evaluation target date: add h months to current date
                     target_date=forecast_date+pd.DateOffset(months=h)
                     #get actual value from df_yoy
                     actual_val= df_yoy.loc[target_date, yoy_col] 

                     #reconstruct the forecasted YoY
                     if h==12:
                        #for 12m ahead, the qrf target is already yoy
                        preds_dense_yoy= preds_dense[idx:idx+1]
                        preds_plot_yoy= preds_plot[idx:idx+1]
                     else: #if h<12, combine known histaory and model preds
                        months_back=12 - h  #need the change from t-(12-h) to t
                        history_date= forecast_date -pd.DateOffset(months=months_back)
                        #get law log prices
                        p_t=np.log(df_yoy.loc[forecast_date, yoy_raw])
                        p_hist=np.log(df_yoy.loc[history_date, yoy_raw])
                        #calc growth that already happened
                        base_effect= (p_t-p_hist)*100
                        #deannualize the model preds
                        scaling_factor= h/12
                        pred_dense_h_step= preds_dense[idx:idx+1] *scaling_factor
                        pred_plot_h_step= preds_plot[idx:idx+1] *scaling_factor 
                        #combine
                        preds_dense_yoy=base_effect +pred_dense_h_step
                        preds_plot_yoy= base_effect+pred_plot_h_step

                     #flatten to 1D array to fit skew-t distribution later
                     y_fit_data=preds_dense_yoy.flatten()
                     skew_params=fit_skew_t(y_fit_data, eval_quantiles)  #fit skew-t, get params by the 99 points

                     #calc PIT (for plotting later): cdf of actual value under fitted skew-t
                     pit_val =nct.cdf(actual_val, df=skew_params[0], nc=skew_params[1], loc=skew_params[2], scale=skew_params[3])
                     #calc step-specific CRPS
                     parametric_crps=calculate_crps(actual_val, skew_params)
                     #want to calc empirical crpsto see how much smoothing the skew-t fit changed the result
                     empirical_crps= qrf_crps_scorer([actual_val], preds_dense_yoy, eval_quantiles)
                     #make dic of result
                     result={'Date':forecast_date, 'Target_date': target_date, 'Actual': actual_val, 'Forecast_median': preds_plot_yoy[0,2],'q05': preds_plot_yoy[0,0],
                             'q16':preds_plot_yoy[0,1], 'q84': preds_plot_yoy[0, 3],'q95': preds_plot_yoy[0, 4], 'Empirical_CRPS': empirical_crps, 'Parametric_CRPS': parametric_crps,
                             'Skewt_df': skew_params[0], 'Skewt_nc': skew_params[1], 'Skewt_loc': skew_params[2], 'Skewt_scale': skew_params[3], 'PIT': pit_val}
                    #append
                     recursive_preds.append(result)
                #advance window 1quarter
                current_idx= next_step_idx
            
            #save and evaluate final recursive results
            results_df= pd.DataFrame(recursive_preds)
            results_df.set_index('Date', inplace=True)
            save_name=f"Results/Data_experiments/{config['experiment_name']}_{target_name}_{h}m.csv"
            results_df.to_csv(save_name)



#run the model 
if __name__=="__main__":
    parser= argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args= parser.parse_args()    
    with open(args.config, 'r') as f:
        conf=yaml.safe_load(f)        
    run_experiment(conf)