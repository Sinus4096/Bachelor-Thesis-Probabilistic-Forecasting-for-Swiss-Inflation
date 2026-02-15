from pathlib import Path
import sys
import pandas as pd
import numpy as np
from quantile_forest import RandomForestQuantileRegressor
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler
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
from Scripts.Utils.PCA import get_pca, make_factor_features_time_safe
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
                #at current_idx, last observable target is at current_idx -h
                last_trainable_idx = current_idx - h  
                #safety sheck for enough data
                if last_trainable_idx< 0:
                    current_idx+= 1
                    continue
                #define window
                train_indices = range(0, last_trainable_idx + 1)

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
                X_train_raw = X_train.copy()
                X_test_raw = X_test.copy()
                #fit standard scaler recursively
                scaler = StandardScaler()
                X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
                X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
                if use_pca_factors:
                    #decide PCA block vs kept columns (AR + seasonals)
                    pca_cols, keep_cols = get_pca(df_columns=X_train.columns, target_cols_to_drop=target_cols_to_drop, target_name=target_name, config=config)
                    #fit scaler+PCA on train
                    X_train, X_test, pca_info = make_factor_features_time_safe(X_train=X_train,  X_test=X_test, pca_cols=pca_cols,keep_cols=keep_cols,
                    config=config, forecast_date=forecast_date, target_name=target_name, h=h, top_k=5,)
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
                #add linear part if residual forecasting
                preds_plot+= 0
                preds_dense+= 0
                #calculate shapley values for the qrf tree part
                shap_tree= shap_values(model, X_test, X_train=X_train,model_type='tree')
                #initialize dict for combined shap values if residual forecasting, if no residual, just the tree one
                shap_combined= shap_tree.copy()           
                #check if target date exists in df_yoy (if not, cannot evaluate)
                if target_date in df_yoy.index:
                        
                    #get actual value from df_yoy
                    actual_val= df_yoy.loc[target_date, yoy_col] 
                    if forecast_method=='direct' or h==12:
                        #direct forecast or 12m ahead: use qrf preds directly
                        preds_dense_yoy= preds_dense
                        preds_plot_yoy= preds_plot
                        #base effect and scaling for shapley values
                        base_effect = 0.0
                        scaling_factor = 1.0
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
                    #initialize dict to store shap values for this forecast
                    final_shap={}
                    #apply scaling to shap values if 
                    final_shap['Shap_Base_Effect'] = base_effect
                    #loop through the combined shap values and apply scaling to the tree part (linear part already in original units)
                    for k, v in shap_combined.items():
                        final_shap[k]=v*scaling_factor
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
                    #add shapley values to the result dictionary
                    result.update(final_shap)
                    #append
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