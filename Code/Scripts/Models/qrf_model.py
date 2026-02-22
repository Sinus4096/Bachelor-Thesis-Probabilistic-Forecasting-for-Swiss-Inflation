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
    data_filename=config['data'].get('data_file', 'data_stationary_bvar.csv')
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

                #separate X and Y
                X_slice= df.drop(columns=target_cols_to_drop) #drop all cols starting with target_
                Y_slice= df[target_col]
                #define train and test set
                X_train= X_slice.iloc[train_indices].copy()  
                Y_train= Y_slice.iloc[train_indices].copy()
                X_test= X_slice.iloc[[current_idx]].copy()  #test input is features available today
                lp_train = None
                lp_test = None
                
                if 'Linear_Pred' in X_train.columns:
                    # Save the column
                    lp_train = X_train['Linear_Pred'].copy()
                    lp_test = X_test['Linear_Pred'].copy()
                    
                    # Remove it from X_train/X_test so it doesn't get Scaled or PCA'd
                    X_train = X_train.drop(columns=['Linear_Pred'])
                    X_test = X_test.drop(columns=['Linear_Pred'])
                #don't use df for testing /evaluating but the  yoy changes
                
                #only drop NAN's for the training set: test might have NAN's in the end-> inference
                Y_train= Y_train.dropna()
                X_train= X_train.loc[Y_train.index]
                if lp_train is not None:
                    lp_train = lp_train.loc[Y_train.index]
                
                if use_pca_factors:
                    #decide PCA block vs kept columns (AR + seasonals)
                    pca_cols, keep_cols = get_pca(df_columns=X_train.columns, target_cols_to_drop=target_cols_to_drop, target_name=target_name, config=config)
                    # FIRST iteration: fit PCA
                    if pca_bundle_fixed is None:
                        X_train, X_test, pca_bundle_fixed = make_factor_features_time_safe(X_train=X_train, X_test=X_test, pca_cols=pca_cols, keep_cols=keep_cols, config=config, forecast_date=forecast_date, target_name=target_name, h=h,  top_k=5, pca_bundle=None )

                    # LATER iterations: reuse
                    else:
                        X_train, X_test, _ = make_factor_features_time_safe( X_train=X_train, X_test=X_test, pca_cols=pca_bundle_fixed["pca_cols"], keep_cols=pca_bundle_fixed["keep_cols"], config=config, forecast_date=forecast_date, target_name=target_name, h=h, top_k=5, pca_bundle=pca_bundle_fixed)
                                    
                if lp_train is not None:
                    # Concatenate back to the processed DataFrames
                    X_train['Linear_Pred'] = lp_train
                    X_test['Linear_Pred'] = lp_test
                #if configured to use residual forecasting
                if use_lin_feat:
                    # Check if current X_test has a valid Linear_Pred
                    if pd.isna(X_test['Linear_Pred'].values[0]):
                        # Fallback if the linear model didn't produce a prediction for today
                        X_test['Linear_Pred'] = 0 
                    
                    # Impute training NaNs
                    X_train['Linear_Pred'] = X_train['Linear_Pred'].fillna(0)
                #use final params determined by which model use
                model_args=final_params.copy()
                #ensure reproducibility
                model_args['random_state']=42
                #train model
                model= RandomForestQuantileRegressor(**model_args)
                model.fit(X_train, Y_train)
                #predict key quantiles for evaluation and plotting
                plot_quantiles=[0.05, 0.16, 0.50, 0.84, 0.95]    
                preds_plot=model.predict(X_test, quantiles=list(plot_quantiles))    #pre safe the predictions
                
                #predict dense grid for CRPS and fan charts
                eval_quantiles= np.linspace(0.01, 0.99, 99)
                preds_dense = model.predict(X_test, quantiles=list(eval_quantiles))
                
                #direct target eval
                #--------------------
                actual_direct = df.loc[forecast_date, target_col]

                if pd.isna(actual_direct):
                    current_idx += 1
                    continue

                # fit skew-t to model distribution
                y_fit_direct = preds_dense.flatten()
                skew_params_direct = fit_skew_t(y_fit_direct, eval_quantiles)

                # parametric CRPS (MAIN METRIC)
                crps_direct = calculate_crps(actual_direct, skew_params_direct)
                #to see effect of fitting and smoothing, also calc empirical crps:
                crps_direct_empirical= calculate_crps_quantile([actual_direct], preds_dense, eval_quantiles)

                # PIT
                pit_direct = nct.cdf(
                    actual_direct,
                    skew_params_direct[0],
                    skew_params_direct[1],
                    loc=skew_params_direct[2],
                    scale=skew_params_direct[3],
                )

                rmse_direct = calculate_rmse(actual_direct, preds_plot[0,2])
                #calculate shapley values for the qrf tree part
                shap_tree= shap_values(model, X_test, X_train=X_train,model_type='tree')
                #initialize dict for combined shap values if residual forecasting, if no residual, just the tree one
                shap_combined= shap_tree.copy()   
                final_shap = {f"SHAP_{k}": float(np.asarray(v).squeeze()) for k, v in shap_combined.items()}        
                #check if target date exists in df_yoy (if not, cannot evaluate)
                if target_date not in df_yoy.index:
                    current_idx += 1
                    continue
                
                else:
                    # =========================
                    # (B) EX-POST YOY (PLOTS ONLY)
                    # =========================
                    T = target_date

                    if (T not in df_yoy.index) or (forecast_date not in df_yoy.index):
                        current_idx += 1
                        continue

                    actual_yoy = df_yoy.loc[T, yoy_col]
                    if pd.isna(actual_yoy):
                        current_idx += 1
                        continue

                    lower = T - pd.DateOffset(months=12)
                    if lower not in df_yoy.index:
                        current_idx += 1
                        continue

                    p_t   = np.log(df_yoy.loc[forecast_date, yoy_raw])   # ex-post anchor
                    p_low = np.log(df_yoy.loc[lower, yoy_raw])

                    base_effect_expost = 100.0 * (p_t - p_low)

                    scaling = h / 12.0
                    preds_plot_yoy_expost  = base_effect_expost + preds_plot * scaling
                    preds_dense_yoy_expost = base_effect_expost + preds_dense * scaling


                    # =========================
                    # (C) TIME-SAFE YoY CRPS (SNB comparable)
                    # =========================
                    pub_lag = 2
                    t_known = forecast_date - pd.DateOffset(months=pub_lag)
                    lower_known = T - pd.DateOffset(months=12)

                    crps_yoy_timesafe_parametric = np.nan

                    if (t_known in df_yoy.index) and (lower_known in df_yoy.index):

                        p_known = np.log(df_yoy.loc[t_known, yoy_raw])
                        p_low   = np.log(df_yoy.loc[lower_known, yoy_raw])

                        base_effect_timesafe = 100.0 * (p_known - p_low)

                        preds_dense_yoy_timesafe = base_effect_timesafe + preds_dense * scaling

                        # skew-t fit (IMPORTANT — same as BVAR)
                        skew_params_yoy = fit_skew_t(preds_dense_yoy_timesafe.flatten(), eval_quantiles)

                        crps_yoy_timesafe_parametric = calculate_crps(actual_yoy, skew_params_yoy)
                    #make dic of result
                    result = {
                        'Date': forecast_date,
                        'Target_date': target_date,

                        # (A) main evaluation
                        'Actual_direct': actual_direct,
                        'Forecast_median_direct': preds_plot[0, 2],
                        'CRPS_direct_parametric': crps_direct,
                        'CRPS_direct_empirical': crps_direct_empirical,
                        'RMSE_direct': rmse_direct,
                        'PIT': pit_direct,

                        # (B) ex-post YoY for plots
                        'Actual_YoY': actual_yoy,
                        'Forecast_median_YoY': preds_plot_yoy_expost[0, 2],
                        'q05_YoY': preds_plot_yoy_expost[0, 0],
                        'q16_YoY': preds_plot_yoy_expost[0, 1],
                        'q84_YoY': preds_plot_yoy_expost[0, 3],
                        'q95_YoY': preds_plot_yoy_expost[0, 4],

                        # (C) optional SNB-comparable-ish metric (time-safe, no bridge)
                        'CRPS_YoY_timesafe': crps_yoy_timesafe_parametric,
                    }
                    result.update(final_shap)
                    recursive_preds.append(result)
                #advance window 1month
                current_idx+=1
            
            #save and evaluate final recursive results
            results_df= pd.DataFrame(recursive_preds)
            results_df.set_index('Date', inplace=True)
            save_name=f"Results/Data_experiments_qrf2/{config['experiment_name']}_{target_name}_{h}m.csv"
            results_df.to_csv(save_name)



#run the model 
if __name__=="__main__":
    parser= argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args= parser.parse_args()    
    with open(args.config, 'r') as f:
        conf=yaml.safe_load(f)        
    run_experiment(conf)