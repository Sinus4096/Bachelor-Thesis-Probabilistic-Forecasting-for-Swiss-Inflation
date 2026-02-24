from pathlib import Path
import sys
import pandas as pd
import numpy as np
from quantile_forest import RandomForestQuantileRegressor
import yaml
import argparse
from scipy.stats import nct



#get path for utils
current_dir=Path(__file__).resolve().parent
#get project root
scripts_root = current_dir.parent.parent   
sys.path.insert(0, str(scripts_root))

#import needed utils
from Scripts.Utils.metrics import calculate_crps, calculate_rmse, calculate_crps_quantile, shap_values
from Scripts.Utils.density_fitting import fit_skew_t
from Scripts.Utils.qrf_utils import generate_linear_feature_oof, get_pca, make_factor_features_time_safe


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
    use_lin_feat = config['model'].get('use_linear_features', False)
    #get forecast method from config file
    forecast_method=config['model'].get('forecast_method', 'reconstruct')
    #get whether will use pca factors or not
    use_pca_factors = bool(config.get("model", {}).get("use_pca_factors", False))
    #to match training start of the bvar:
    training_offset=14

    #iterate through all targets 
    for target_name in targets:
        #iterate through all horizons defined in script 03
        for h in horizons:
            #just once pca to get interpretability
            pca_bundle_fixed = None
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
            target_cols_to_drop= [col for col in df.columns if 'target_' in col]   #don't want target variable in X later

            final_params= config['model']['params']
            
            #recursive out-of-sample predictions
            recursive_preds = []    #initialize storage for out-of-sample predictions
            #start time loop at eval_start_date-> get index location of eval_start_date 
            requested_start_idx = df.index.get_loc(eval_start_date)
            if isinstance(requested_start_idx, slice):
                requested_start_idx = requested_start_idx.start
            start_idx= max(requested_start_idx, training_offset)  #choose startidx to match with bvar
            current_idx= start_idx

            if isinstance(start_idx, slice):    #if get_loc returns slice->handle
                start_idx= start_idx.start
            total_rows= len(df) #get length of the original df
            if use_lin_feat:
                linear_preds = generate_linear_feature_oof(df=df, target_col=target_col, target_cols_to_drop=target_cols_to_drop, 
                    h=h, config=config, use_pca=use_pca_factors, window_size=120, min_train=40)                
                df['Linear_Pred'] = linear_preds  #add the linear features
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
                pub_lag = 2  #have to wait 2 months for the target to be available

                #at current_idx, last observable target is at current_idx -h
                last_trainable_idx = current_idx - h- pub_lag
                #safety sheck for enough data
                if last_trainable_idx< 0:
                    current_idx+= 1
                    continue
                #define window matching with bvar
                train_indices = range(training_offset, last_trainable_idx + 1)
                #drop all cols starting with target_ to isolate features
                X_slice= df.drop(columns=target_cols_to_drop) 
                #isolate specific target column for estimation
                Y_slice= df[target_col]
                #define train and test set
                #slice features for training window
                X_train= X_slice.iloc[train_indices].copy()  
                #slice target for training window
                Y_train= Y_slice.iloc[train_indices].copy()
                #test input is features available today
                X_test= X_slice.iloc[[current_idx]].copy()  
                #initialize holders for linear predictions
                lp_train= None
                lp_test= None

                #check if linear predictions exist in feature set
                if 'Linear_Pred' in X_train.columns:
                    #save linear prediction column for train
                    lp_train= X_train['Linear_Pred'].copy()
                    #save linear prediction column for test
                    lp_test= X_test['Linear_Pred'].copy()                    
                    #remove from feature matrices to avoid scaling or pca interference
                    X_train= X_train.drop(columns=['Linear_Pred'])
                    #remove from test matrix
                    X_test= X_test.drop(columns=['Linear_Pred'])

                #only drop nans for training set to ensure valid estimation
                Y_train= Y_train.dropna()
                #align features with cleaned target index
                X_train= X_train.loc[Y_train.index]
                #align linear predictions if they exist
                if lp_train is not None:
                    lp_train= lp_train.loc[Y_train.index]
                #handle dimensionality reduction if configured
                if use_pca_factors:
                    #decide pca block vs kept columns like ar or seasonals
                    pca_cols, keep_cols= get_pca(df_columns=X_train.columns, target_cols_to_drop=target_cols_to_drop, target_name=target_name, config=config)
                    #first iteration: fit pca on current window
                    if pca_bundle_fixed is None:
                        #generate factor features and store fit in bundle
                        X_train, X_test, pca_bundle_fixed= make_factor_features_time_safe(X_train=X_train, X_test=X_test, pca_cols=pca_cols, keep_cols=keep_cols, config=config, forecast_date=forecast_date, target_name=target_name, h=h,  top_k=5, pca_bundle=None )

                    #later iterations: reuse existing pca fit for stability
                    else:
                        #transform features using stored pca bundle
                        X_train, X_test, _= make_factor_features_time_safe( X_train=X_train, X_test=X_test, pca_cols=pca_bundle_fixed["pca_cols"], keep_cols=pca_bundle_fixed["keep_cols"], config=config, forecast_date=forecast_date, target_name=target_name, h=h, top_k=5, pca_bundle=pca_bundle_fixed)
                                
                #re-attach linear predictions if extracted earlier
                if lp_train is not None:
                    #concatenate back to processed training dataframe
                    X_train['Linear_Pred']= lp_train
                    #concatenate back to processed test dataframe
                    X_test['Linear_Pred']= lp_test

                #logic for residual forecasting models
                if use_lin_feat:
                    #check if current test row has valid linear model input
                    if pd.isna(X_test['Linear_Pred'].values[0]):
                        #set to zero if linear model failed for current step
                        X_test['Linear_Pred']= 0 
                    
                    #fill missing training values with zero for model stability
                    X_train['Linear_Pred']= X_train['Linear_Pred'].fillna(0)

                #setup model arguments from config
                model_args= final_params.copy()
                #ensure results are reproducible across runs
                model_args['random_state']= 42
                #initialize quantile random forest model
                model= RandomForestQuantileRegressor(**model_args)
                #fit qrf on current training data
                model.fit(X_train, Y_train)
                #define key quantiles for plots and evaluation
                plot_quantiles= [0.05, 0.16, 0.50, 0.84, 0.95]    
                #generate predictions for plotting quantiles
                preds_plot= model.predict(X_test, quantiles=list(plot_quantiles))    

                #predict dense grid for crps and fan charts
                #setup grid of 99 percentiles
                eval_quantiles= np.linspace(0.01, 0.99, 99)
                #generate dense predictive distribution
                preds_dense= model.predict(X_test, quantiles=list(eval_quantiles))

                #direct target eval
                #--------------------
                #get realized value for current forecast date
                actual_direct= df.loc[forecast_date, target_col]

                #skip iteration if actual value is not yet available
                if pd.isna(actual_direct):
                    #increment index to next month
                    current_idx+= 1
                    #continue to next loop
                    continue
                #fit skew-t distribution to model quantiles
                y_fit_direct= preds_dense.flatten()     #flatten predictive grid for fitting
                skew_params_direct= fit_skew_t(y_fit_direct, eval_quantiles) #estimate distribution parameters
                #calculate parametric crps as main performance metric
                crps_direct= calculate_crps(actual_direct, skew_params_direct)
                #calculate empirical crps to see effect of smoothing
                crps_direct_empirical= calculate_crps_quantile([actual_direct], preds_dense, eval_quantiles)

                #calculate probability integral transform for calibration check
                pit_direct= nct.cdf(actual_direct, skew_params_direct[0], skew_params_direct[1], loc=skew_params_direct[2], scale=skew_params_direct[3])
                #calculate root mean squared error for point forecast
                rmse_direct= calculate_rmse(actual_direct, preds_plot[0,2])
                #calculate shapley values to interpret tree model
                shap_tree= shap_values(model, X_test, X_train=X_train,model_type='tree')
                #init combined shap if using residual forecasting logic
                shap_combined= shap_tree.copy()   
                #format shap results into dictionary with prefix
                final_shap= {f"SHAP_{k}": float(np.asarray(v).squeeze()) for k, v in shap_combined.items()}        
                #verify target date exists in yoy dataframe
                if target_date not in df_yoy.index:
                    #increment and skip if no yoy data
                    current_idx+= 1
                    continue
                else:
                    #yoy for plotting
                    #-------------------------
                    #set target date for yoy mapping
                    T= target_date
                    #check index for date t
                    if T not in df_yoy.index:
                        current_idx+= 1
                        continue

                    #get yoy realization from data
                    actual_yoy= df_yoy.loc[T, yoy_col]
                    #skip if realization is missing
                    if pd.isna(actual_yoy):
                        current_idx+= 1
                        continue
                    #calculate time scaling factor for horizon
                    scaling= h /12.0

                    #handle special argarch case for 12m horizon
                    if h ==12:
                        #no base effect for direct yoy model
                        base_effect_expost= 0.0
                        #yoy preds are just direct preds
                        preds_plot_yoy_expost= preds_plot.copy()
                        #yoy dense is just direct dense
                        preds_dense_yoy_expost= preds_dense.copy()
                    else:
                        #reconstruct yoy from direct log changes
                        lower= T-pd.DateOffset(months=12)

                        #verify anchor dates exist in dataset
                        if (forecast_date not in df_yoy.index) or (lower not in df_yoy.index):
                            current_idx+= 1
                            continue
                        #get log levels for anchor points
                        p_t= np.log(df_yoy.loc[forecast_date, yoy_raw])  
                        p_low= np.log(df_yoy.loc[lower, yoy_raw])

                        #calculate base effect from known historical levels
                        base_effect_expost= 100.0 * (p_t - p_low)

                        #reconstruct yoy plot quantiles
                        preds_plot_yoy_expost= base_effect_expost + preds_plot * scaling
                        #reconstruct yoy dense distribution
                        preds_dense_yoy_expost= base_effect_expost + preds_dense * scaling


                    #time-safe yoy crps
                    #--------------------------
                    #set publication lag for snb comparison
                    pub_lag= 2
                    #get latest available data point at forecast origin
                    t_known= forecast_date - pd.DateOffset(months=pub_lag)
                    #init holders for time-safe metrics
                    crps_yoy_timesafe_parametric= np.nan
                    crps_yoy_timesafe_empirical= np.nan
                    #handle argarch-style h12
                    if h == 12:
                        #no reconstruction needed for direct yoy
                        preds_dense_yoy_timesafe= preds_dense.copy()
                    else:
                        #find anchor point for time-safe reconstruction
                        lower_known= T-pd.DateOffset(months=12)
                        #verify data available at publication lag
                        if (t_known in df_yoy.index) and (lower_known in df_yoy.index):
                            #get historical log levels
                            p_known= np.log(df_yoy.loc[t_known, yoy_raw])
                            p_low= np.log(df_yoy.loc[lower_known, yoy_raw])
                            #calculate base effect available at origin t
                            base_effect_timesafe= 100.0 * (p_known - p_low)
                            #reconstruct predictive distribution without using t-0 data
                            preds_dense_yoy_timesafe= base_effect_timesafe + preds_dense * scaling
                        else:
                            #mark as none if data not available
                            preds_dense_yoy_timesafe= None

                    #evaluate reconstructed yoy distribution
                    if preds_dense_yoy_timesafe is not None:
                        #format quantiles for scoring function
                        yoy_q= np.asarray(preds_dense_yoy_timesafe).reshape(1, -1)

                        #calculate empirical score from reconstructed quantiles
                        crps_yoy_timesafe_empirical= float(np.mean(calculate_crps_quantile([actual_yoy], yoy_q, eval_quantiles)))
                        #fit distribution to reconstructed quantiles
                        skew_params_yoy= fit_skew_t(yoy_q.flatten(), eval_quantiles)
                        #calculate parametric score for yoy
                        crps_yoy_timesafe_parametric= float(calculate_crps(actual_yoy, skew_params_yoy))

                    #package all metrics into result dictionary
                    result= {'Date': forecast_date, 'Target_date': target_date, 'Actual_direct': actual_direct, 'Forecast_median_direct': preds_plot[0, 2],
                        'CRPS_direct_parametric': crps_direct, 'CRPS_direct_empirical': crps_direct_empirical, 'RMSE_direct': rmse_direct,
                        'PIT_direct': pit_direct, 'df_skewt_direct': float(skew_params_direct[0]), 'nc_skewt_direct': float(skew_params_direct[1]),
                        'loc_skewt_direct': float(skew_params_direct[2]),'scale_skewt_direct': float(skew_params_direct[3]), 'Actual_YoY': actual_yoy,
                        'Forecast_median_YoY': preds_plot_yoy_expost[0, 2], 'q05_YoY': preds_plot_yoy_expost[0, 0], 'q16_YoY': preds_plot_yoy_expost[0, 1],
                        'q84_YoY': preds_plot_yoy_expost[0, 3], 'q95_YoY': preds_plot_yoy_expost[0, 4], 'BaseEffect_YoY_expost': float(base_effect_expost), 'CRPS_YoY_timesafe_parametric': crps_yoy_timesafe_parametric,
                        'CRPS_YoY_timesafe_empirical': crps_yoy_timesafe_empirical}
                    #append shapley values to result
                    result.update(final_shap)
                    #store result in recursive list
                    recursive_preds.append(result)
                #advance rolling window one month
                current_idx+= 1

                #save and evaluate final recursive results table
                results_df= pd.DataFrame(recursive_preds)   #convert list of results to dataframe
                #set forecast origin as index
                results_df.set_index('Date', inplace=True)
                #set save path for experiment/target/horizon
                save_name= f"Results/Data_experiments_qrf2/{config['experiment_name']}_{target_name}_{h}m.csv"
                #export to csv for plotting
                results_df.to_csv(save_name)



#run the model 
if __name__=="__main__":
    parser= argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args= parser.parse_args()    
    with open(args.config, 'r') as f:
        conf=yaml.safe_load(f)        
    run_experiment(conf)