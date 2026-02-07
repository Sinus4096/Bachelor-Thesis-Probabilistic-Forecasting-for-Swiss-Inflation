from pathlib import Path
import sys
import pandas as pd
import numpy as np
from quantile_forest import RandomForestQuantileRegressor

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler
import yaml
import argparse
from scipy.stats import nct
from sklearn.linear_model import Ridge
#get path for utils
current_dir=Path(__file__).resolve().parent
#get project root
scripts_root = current_dir.parent.parent   
sys.path.insert(0, str(scripts_root))

#import needed utils
from Scripts.Utils.metrics import qrf_crps_scorer, calculate_crps, calculate_rmse, calculate_crps_quantile
from Scripts.Utils.density_fitting import fit_skew_t


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
    #get which data to use from config file
    data_filename=config['data'].get('data_file', 'data_stationary.csv')
    #load data
    project_root=current_dir.parent.parent
    #get path to selected data
    data_path =project_root /"Data"/ "Cleaned_Data"/data_filename
    df =pd.read_csv(data_path, index_col='Date', parse_dates=True)
    data_yoy_path=project_root/ "Data"/ "Cleaned_Data"/"data_yoy.csv"
    df_yoy=pd.read_csv(data_yoy_path, index_col='Date', parse_dates=True)

    #get target variables and forecast horizon from the config file
    targets=config['data']['targets']
    horizons=config['data']['horizons']
    #SNB forecasts once per quarter: in march, june, september, december
    snb_months=[3, 6, 9, 12]
    #set recursive (out-of-sampe) prediction windos (->when stop training and update after how many months)
    eval_start_date= pd.Timestamp(config['data']['eval_start_date'])

    #for residual forecastin (in config file need residuals)
    use_residuals = config['model'].get('use_residual_forecasting', False)
    #get forecast method from config file
    forecast_method = config['model'].get('forecast_method', 'reconstruct')
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
                #get date of current index
                current_date= df.index[current_idx]
                #define forecast window:
                forecast_date= current_date
                target_date= forecast_date + pd.DateOffset(months=h)
                #if not a forecast month: skip to next
                if current_date.month not in snb_months:
                    current_idx+= 1
                    continue
                #training set ranges from beginning up to the current index
                train_indices= range(0, current_idx)    
                
                if current_idx >= total_rows:
                    break #get out if reached the last row already

                #define ssubdf of original df up until next forecast step
                df_slice = df.iloc[:current_idx + 1].copy()

                #separate X and Y
                X_slice= df_slice.drop(columns=target_cols_to_drop) #drop all cols starting with target_
                Y_slice= df_slice[target_col]
                #define train and test set
                X_train= X_slice.iloc[train_indices]
                Y_train = Y_slice.iloc[train_indices]
                X_test = X_slice.iloc[[current_idx]]
                #don't use df for testing /evaluating but the  yoy changes
                #only drop NAN's for the training set: test might have NAN's in the end-> inference
                Y_train= Y_train.dropna()
                X_train= X_train.loc[Y_train.index]


                #if configured to use residual forecasting
                if use_residuals:
                    #scale Data 
                    scaler= StandardScaler()
                    X_train_scaled=scaler.fit_transform(X_train)
                    X_test_scaled= scaler.transform(X_test)
                    #define rolling window for structural breaks: 5y=60months
                    window_size=60
                    #ensure we have enough data for splits, else reduce splits
                    n_splits_dynamic =min(5, len(X_train) - 2)
                    if n_splits_dynamic > 1:
                        #TimeSeriesSplit with max_train_size creates rolling effect, validate 1 step ahead
                        tscv_rolling = TimeSeriesSplit(n_splits=n_splits_dynamic, test_size=1, max_train_size=window_size)
                        #grid search for best alpha
                        ridge_params= {'alpha': [0.1, 1.0, 10.0, 50.0, 100.0, 500.0]}
                        grid_search = GridSearchCV(Ridge(),ridge_params,cv=tscv_rolling,scoring='neg_mean_squared_error',n_jobs=-1)
                        # Even if CV selects the best alpha, we shouldn't fit on 
                        # data from 20 years ago if there is a break.
                        # We limit the training data to the rolling window for the fit.
                        if len(X_train_scaled) > window_size:
                            X_train_recent = X_train_scaled[-window_size:]
                            Y_train_recent = Y_train.iloc[-window_size:]
                        else:
                            X_train_recent = X_train_scaled
                            Y_train_recent = Y_train
                    
                        # Fit grid on recent data (or full, but CV logic dictates preference)
                        # Actually, typically we fit Grid on recent data to find Alpha
                        grid_search.fit(X_train_recent, Y_train_recent)
                        best_ridge = grid_search.best_estimator_
                        
                    else:
                        # Fallback if too little data for CV
                        best_ridge = Ridge(alpha=1.0)
                        best_ridge.fit(X_train_scaled, Y_train)

                    # 3. Predict Residuals
                    # Train residuals (in-sample) - Apply model to full history for QRF
                    train_linear_preds = best_ridge.predict(X_train_scaled)
                    Y_train_effective = Y_train - train_linear_preds
                    
                    # Test prediction (linear part)
                    test_linear_preds = best_ridge.predict(X_test_scaled)
                    #fit ridge regression to get residuals
                    ridge_baseline= Ridge(alpha=1.0) 
                    ridge_baseline.fit(X_train, Y_train)
                    #get residuals for qrf training
                    train_linear_preds=ridge_baseline.predict(X_train)  #predict with linear model
                    Y_train_effective= Y_train-train_linear_preds                    
                    #get linear forecast for test set
                    test_linear_preds= ridge_baseline.predict(X_test)
                
                else: #normal qrf forecasting
                    Y_train_effective= Y_train
                    test_linear_preds=np.zeros(len(X_test))  #no linear effect to add later


                #use final params determined by which model use
                model_args=final_params.copy()
                #ensure reproducibility
                model_args['random_state']=42
                #train model
                model= RandomForestQuantileRegressor(**model_args)
                model.fit(X_train, Y_train_effective)
                #predict key quantiles for evaluation and plotting
                plot_quantiles=[0.05, 0.16, 0.50, 0.84, 0.95]    
                preds_plot=model.predict(X_test, quantiles=list(plot_quantiles))    #pre safe the predictions
                #get linear additve part if residual forecasting
                linear_add = test_linear_preds.reshape(-1, 1)
                #predict dense grid for CRPS and fan charts
                eval_quantiles= np.linspace(0.01, 0.99, 99)
                preds_dense = model.predict(X_test, quantiles=list(eval_quantiles))
                #add linear part if residual forecasting
                preds_plot+= linear_add
                preds_dense+= linear_add
                
                #check if target date exists in df_yoy (if not, cannot evaluate)
                if target_date in df_yoy.index:
                        
                     #get actual value from df_yoy
                     actual_val= df_yoy.loc[target_date, yoy_col] 
                     if forecast_method=='direct' or h==12:
                         #direct forecast or 12m ahead: use qrf preds directly
                         preds_dense_yoy= preds_dense
                         preds_plot_yoy= preds_plot
                     #else reconstruct the forecasted YoY                     
                     else: #if h<12, combine known histaory and model preds
                        months_back=12 - h  #need the change from t-(12-h) to t
                        history_date= forecast_date -pd.DateOffset(months=months_back)
                        #check if history date exists
                        if history_date not in df_yoy.index:
                            #cannot reconstruct if no history date-> skip
                            current_idx+=1
                            continue
                        #get law log prices
                        p_t=np.log(df_yoy.loc[forecast_date, yoy_raw])
                        p_hist=np.log(df_yoy.loc[history_date, yoy_raw])
                        #calc growth that already happened
                        base_effect= (p_t-p_hist)*100
                        #deannualize the model preds
                        scaling_factor= h/12
                        pred_dense_h_step= preds_dense *scaling_factor
                        pred_plot_h_step= preds_plot *scaling_factor 
                        #combine
                        preds_dense_yoy=base_effect +pred_dense_h_step
                        preds_plot_yoy= base_effect+pred_plot_h_step
                     #calc rmse to tell whether model that is better in probabilistic terms also better in point forecast terms (call on median), average later
                     sq_error= calculate_rmse(actual_val, preds_plot_yoy[0,2])
                     #flatten to 1D array to fit distribution later
                     y_fit_data=preds_dense_yoy.flatten()
                     

                     skew_params=fit_skew_t(y_fit_data, eval_quantiles)  #fit skew-t, get params by the 99 points
                         
                     #calc PIT (for plotting later): cdf of actual value under fitted skew-t
                     pit_val= nct.cdf(actual_val, skew_params[0], skew_params[1], loc=skew_params[2], scale=skew_params[3])
                     #calc step-specific CRPS for skew-t
                     parametric_crps=calculate_crps(actual_val, skew_params)
                     #get params for plotting later
                     dist_params= {'df': skew_params[0], 'nc': skew_params[1], 'loc': skew_params[2], 'scale': skew_params[3]}
                     #want to calc empirical crpsto see how much smoothing the skew-t or KDE fit changed the result
                     empirical_crps= calculate_crps_quantile([actual_val], preds_dense_yoy, eval_quantiles)
                     #make dic of result
                     result={'Date':forecast_date, 'Target_date': target_date, 'Actual': actual_val, 'Forecast_median': preds_plot_yoy[0,2],'q05': preds_plot_yoy[0,0],
                             'q16':preds_plot_yoy[0,1], 'q84': preds_plot_yoy[0, 3],'q95': preds_plot_yoy[0, 4], 'Squared_Error': sq_error, 'Empirical_CRPS': empirical_crps, 'Parametric_CRPS': parametric_crps,
                             'df_skewt': dist_params['df'], 'nc_skewt': dist_params['nc'], 'loc_skewt': dist_params['loc'], 'scale_skewt': dist_params['scale'], 'PIT': pit_val}
                    #append
                     recursive_preds.append(result)
                #advance window 1month
                current_idx+=1
            
            #save and evaluate final recursive results
            results_df= pd.DataFrame(recursive_preds)
            results_df.set_index('Date', inplace=True)
            save_name=f"Results/Data_experiments_qrf/{config['experiment_name']}_{target_name}_{h}m.csv"
            results_df.to_csv(save_name)



#run the model 
if __name__=="__main__":
    parser= argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args= parser.parse_args()    
    with open(args.config, 'r') as f:
        conf=yaml.safe_load(f)        
    run_experiment(conf)