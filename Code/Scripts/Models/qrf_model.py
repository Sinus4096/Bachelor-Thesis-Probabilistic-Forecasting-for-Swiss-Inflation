from pathlib import Path
import sys
import pandas as pd
import numpy as np
from quantile_forest import RandomForestQuantileRegressor
import yaml
import argparse
from scipy.stats import nct
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit

#get path for utils
current_dir= Path(__file__).resolve().parent
#define scripts root for imports
scripts_root= current_dir.parent.parent
#insert into system path
sys.path.insert(0, str(scripts_root))

#import needed utils
from Scripts.Utils.metrics import (calculate_crps, calculate_rmse, calculate_crps_quantile, shap_values,)
from Scripts.Utils.density_fitting import fit_skew_t
from Scripts.Utils.qrf_utils import (get_pca, make_factor_features_time_safe, fit_enet_mean_and_residuals)




def run_experiment(config):
    #print current experiment name from config
    print(f"run {config['experiment_name']}")
    #get data filename from config
    data_filename= config["data"].get("data_file", "data_stationary.csv")
    #set project root path
    project_root= current_dir.parent.parent
    #construct full data path
    data_path= project_root / "Data" / "Cleaned_Data" / data_filename
    #load feature data with date index
    df= pd.read_csv(data_path, index_col="Date", parse_dates=True)
    #construct yoy data path
    data_yoy_path= project_root / "Data" / "Cleaned_Data" / "data_yoy.csv"
    #load yoy inflation for reconstruction
    df_yoy= pd.read_csv(data_yoy_path, index_col="Date", parse_dates=True)
    #get target names from config
    targets= config["data"]["targets"]
    #get horizons from config
    horizons= config["data"]["horizons"]
    #get evaluation start date
    eval_start_date= pd.Timestamp(config["data"]["eval_start_date"])
    #define snb policy months
    snb_months= [3, 6, 9, 12]
    #define quantiles for plotting
    plot_quantiles= [0.05, 0.16, 0.50, 0.84, 0.95]
    #define dense grid for distribution fit
    eval_quantiles= np.linspace(0.01, 0.99, 99)
    #set training offset
    training_offset= 14
    #set publication lag
    pub_lag= 2
    #check if residual forecasting enabled
    use_lin_feat= bool(config.get("model", {}).get("use_linear_features", False))
    #check if factor features enabled
    use_pca_factors= bool(config.get("model", {}).get("use_pca_factors", False))
    #load model parameters
    final_params= config["model"]["params"]
    #loop through target variables
    for target_name in targets:
        #loop through forecast horizons
        for h in horizons:
            #reset pca bundle for new horizon
            pca_bundle_fixed= None
            #set column names for headline target
            if target_name == "Headline":
                target_col= f"target_headline_{h}m"
                yoy_col= "Headline"
                yoy_raw= "Headline_level"
            #set column names for core target
            else:
                target_col= f"target_core_{h}m"
                yoy_col= "Core"
                yoy_raw= "Core_level"
            #skip if target missing from data
            if target_col not in df.columns:
                continue
            #identify target columns to drop for x matrix
            target_cols_to_drop= [c for c in df.columns if "target_" in c]
            #find start index in index
            requested_start_idx= df.index.get_loc(eval_start_date)
            #handle slice indexing
            if isinstance(requested_start_idx, slice):
                requested_start_idx= requested_start_idx.start
            #ensure start respects offset
            start_idx= max(requested_start_idx, training_offset)
            #get total observations
            total_rows= len(df)
            #init holder for recursive preds
            recursive_preds= []
            #set start index for loop
            current_idx= start_idx
            #main rolling window loop
            while current_idx < total_rows:
                #get current origin date
                current_date= df.index[current_idx]
                #label forecast date
                forecast_date= current_date
                #calculate target realization date
                target_date= forecast_date + pd.DateOffset(months=h)
                #only forecast on snb months
                if current_date.month not in snb_months:
                    current_idx+= 1
                    continue
                #find last available training data point
                last_trainable_idx= current_idx - h - pub_lag
                #skip if window too small
                if last_trainable_idx < 0:
                    current_idx+= 1
                    continue
                #define expanding window range
                train_indices= range(training_offset, last_trainable_idx + 1)
                #isolate feature matrix
                X_slice= df.drop(columns=target_cols_to_drop)
                #isolate target vector
                Y_slice= df[target_col]
                #slice training features
                X_train= X_slice.iloc[train_indices].copy()
                #slice training targets
                Y_train= Y_slice.iloc[train_indices].copy()
                #slice single test point
                X_test= X_slice.iloc[[current_idx]].copy()
                #drop target nans
                Y_train= Y_train.dropna()
                #align features with targets
                X_train= X_train.loc[Y_train.index]
                #handle factor transformation
                if use_pca_factors:
                    #identify cols for pca vs keeping
                    pca_cols, keep_cols= get_pca(df_columns=X_train.columns, target_cols_to_drop=target_cols_to_drop, target_name=target_name, config=config)
                    #fit factors once if not fixed
                    if pca_bundle_fixed is None:
                        X_train, X_test, pca_bundle_fixed= make_factor_features_time_safe(X_train=X_train, X_test=X_test, pca_cols=pca_cols, keep_cols=keep_cols, config=config, forecast_date=forecast_date, target_name=target_name, h=h, top_k=5, pca_bundle=None)
                    #reuse pca loadings for stability
                    else:
                        X_train, X_test, _= make_factor_features_time_safe(X_train=X_train, X_test=X_test, pca_cols=pca_bundle_fixed["pca_cols"], keep_cols=pca_bundle_fixed["keep_cols"], config=config, forecast_date=forecast_date, target_name=target_name, h=h, top_k=5, pca_bundle=pca_bundle_fixed)
                #handle residual path logic
                if use_lin_feat:
                    #fit linear mean model
                    y_resid_train, mean_test= fit_enet_mean_and_residuals(X_train=X_train, y_train=Y_train, X_test=X_test, h=h, pub_lag=pub_lag)
                    #align feature matrix for residual fit
                    X_train_used= X_train.loc[y_resid_train.index].copy()
                    #set target as residuals
                    Y_train_used= y_resid_train.copy()
                    #calculate medians for imputation
                    meds= X_train_used.median(numeric_only=True)
                    #fill missing values in train
                    X_train_used= X_train_used.fillna(meds).fillna(0.0)
                    #fill missing values in test
                    X_test_used= X_test.fillna(meds).fillna(0.0)
                    #ensure numeric types
                    X_train_used= X_train_used.apply(pd.to_numeric, errors="coerce").fillna(0.0)
                    X_test_used= X_test_used.apply(pd.to_numeric, errors="coerce").fillna(0.0)
                #handle standard path logic
                else:
                    mean_test= 0.0
                    X_train_used= X_train
                    Y_train_used= Y_train
                    X_test_used= X_test
                #copy model arguments
                model_args= final_params.copy()
                #set reproducibility seed
                model_args["random_state"]= 42
                #init qrf model
                model= RandomForestQuantileRegressor(**model_args)
                #fit tree model
                model.fit(X_train_used, Y_train_used)
                #predict plot quantiles
                preds_plot= model.predict(X_test_used, quantiles=list(plot_quantiles))
                #predict dense quantiles
                preds_dense= model.predict(X_test_used, quantiles=list(eval_quantiles))
                #add linear mean back if using residual path
                if use_lin_feat:
                    preds_plot= preds_plot + mean_test
                    preds_dense= preds_dense + mean_test
                #get direct realization
                actual_direct= df.loc[forecast_date, target_col]
                #skip if no actual available
                if pd.isna(actual_direct):
                    current_idx+= 1
                    continue
                #flatten density for fitting
                y_fit_direct= preds_dense.flatten()
                #smooth to skew-t distribution
                skew_params_direct= fit_skew_t(y_fit_direct, eval_quantiles)
                #calc parametric crps
                crps_direct= calculate_crps(actual_direct, skew_params_direct)
                #calc empirical crps
                crps_direct_empirical= calculate_crps_quantile([actual_direct], preds_dense, eval_quantiles)
                #calc pit for calibration
                pit_direct= nct.cdf(actual_direct, skew_params_direct[0], skew_params_direct[1], loc=skew_params_direct[2], scale=skew_params_direct[3])
                #get nct parameters
                df_nct, nc_nct, loc_nct, scale_nct= skew_params_direct
                #calc log predictive density
                logpdf_direct= float(nct.logpdf(actual_direct, df_nct, nc_nct, loc=loc_nct, scale=scale_nct))
                #assign logs score
                logS_direct= logpdf_direct
                #calc rmse for point forecast
                rmse_direct= calculate_rmse(actual_direct, preds_plot[0, 2])
                #calc tree based shap values
                shap_tree= shap_values(model, X_test_used, X_train=X_train_used, model_type="tree")
                #format shap dictionary
                final_shap= {f"SHAP_{k}": float(np.asarray(v).squeeze()) for k, v in shap_tree.items()}
                #skip if target date missing from yoy file
                if target_date not in df_yoy.index:
                    current_idx+= 1
                    continue
                #set target time
                T= target_date
                #get actual yoy inflation
                actual_yoy= df_yoy.loc[T, yoy_col]
                #skip if yoy nan
                if pd.isna(actual_yoy):
                    current_idx+= 1
                    continue
                #set horizon scaling
                scaling= h / 12.0
                #handle 12m horizon reconstruction
                if h == 12:
                    base_effect_expost= 0.0
                    preds_plot_yoy_expost= preds_plot.copy()
                    preds_dense_yoy_expost= preds_dense.copy()
                #handle non-12m reconstruction
                else:
                    lower= T - pd.DateOffset(months=12)
                    if (forecast_date not in df_yoy.index) or (lower not in df_yoy.index):
                        current_idx+= 1
                        continue
                    p_t= np.log(df_yoy.loc[forecast_date, yoy_raw])
                    p_low= np.log(df_yoy.loc[lower, yoy_raw])
                    base_effect_expost= 100.0 * (p_t - p_low)
                    preds_plot_yoy_expost= base_effect_expost + preds_plot * scaling
                    preds_dense_yoy_expost= base_effect_expost + preds_dense * scaling
                #set known time for timesafe comparison
                t_known= forecast_date - pd.DateOffset(months=pub_lag)
                #init holders for yoy metrics
                crps_yoy_timesafe_parametric= np.nan
                crps_yoy_timesafe_empirical= np.nan
                #reconstruct timesafe distribution
                if h == 12:
                    preds_dense_yoy_timesafe= preds_dense.copy()
                else:
                    lower_known= T - pd.DateOffset(months=12)
                    if (t_known in df_yoy.index) and (lower_known in df_yoy.index):
                        p_known= np.log(df_yoy.loc[t_known, yoy_raw])
                        p_low= np.log(df_yoy.loc[lower_known, yoy_raw])
                        base_effect_timesafe= 100.0 * (p_known - p_low)
                        preds_dense_yoy_timesafe= base_effect_timesafe + preds_dense * scaling
                    else:
                        preds_dense_yoy_timesafe= None
                #reset quantile trackers
                q05_yoy_timesafe= np.nan
                q16_yoy_timesafe= np.nan
                q84_yoy_timesafe= np.nan
                q95_yoy_timesafe= np.nan
                median_yoy_timesafe= np.nan
                violation_90_timesafe= np.nan
                upper_violation_95_timesafe= np.nan
                #evaluate yoy distribution if reconstructed
                if preds_dense_yoy_timesafe is not None:
                    yoy_q= np.asarray(preds_dense_yoy_timesafe).reshape(1, -1)
                    crps_yoy_timesafe_empirical= float(np.mean(calculate_crps_quantile([actual_yoy], yoy_q, eval_quantiles)))
                    skew_params_yoy= fit_skew_t(yoy_q.flatten(), eval_quantiles)
                    crps_yoy_timesafe_parametric= float(calculate_crps(actual_yoy, skew_params_yoy))
                    q05_yoy_timesafe= float(np.percentile(preds_dense_yoy_timesafe, 5))
                    q16_yoy_timesafe= float(np.percentile(preds_dense_yoy_timesafe, 16))
                    q84_yoy_timesafe= float(np.percentile(preds_dense_yoy_timesafe, 84))
                    q95_yoy_timesafe= float(np.percentile(preds_dense_yoy_timesafe, 95))
                    median_yoy_timesafe= float(np.median(preds_dense_yoy_timesafe))
                    violation_90_timesafe= int((actual_yoy <q05_yoy_timesafe) or (actual_yoy >q95_yoy_timesafe))
                    upper_violation_95_timesafe= int(actual_yoy> q95_yoy_timesafe)
                #build result dictionary
                result= {"Date": forecast_date, "Target_date": target_date, "Actual_direct": actual_direct,
                    "Forecast_median_direct": preds_plot[0, 2], "CRPS_direct_parametric": crps_direct, "CRPS_direct_empirical": crps_direct_empirical,
                    "LogS_direct": logS_direct, "RMSE_direct": rmse_direct, "PIT_direct": pit_direct,
                    "df_skewt_direct": float(skew_params_direct[0]), "nc_skewt_direct": float(skew_params_direct[1]), "loc_skewt_direct": float(skew_params_direct[2]),
                    "scale_skewt_direct": float(skew_params_direct[3]), "Actual_YoY": actual_yoy, "Forecast_median_YoY": preds_plot_yoy_expost[0, 2],
                    "q05_YoY": preds_plot_yoy_expost[0, 0], "q16_YoY": preds_plot_yoy_expost[0, 1], "q84_YoY": preds_plot_yoy_expost[0, 3],
                    "q95_YoY": preds_plot_yoy_expost[0, 4], "BaseEffect_YoY_expost": float(base_effect_expost), "CRPS_YoY_timesafe_parametric": crps_yoy_timesafe_parametric,
                    "CRPS_YoY_timesafe_empirical": float(crps_yoy_timesafe_empirical) if np.isfinite(crps_yoy_timesafe_empirical) else np.nan,
                    "q05_YoY_timesafe": q05_yoy_timesafe, "q16_YoY_timesafe": q16_yoy_timesafe, "q84_YoY_timesafe": q84_yoy_timesafe,
                    "q95_YoY_timesafe": q95_yoy_timesafe, "Violation90_YoY_timesafe": violation_90_timesafe, "UpperViolation95_YoY_timesafe": upper_violation_95_timesafe}
                #update results with shapley values
                result.update(final_shap)
                #append result to list
                recursive_preds.append(result)
                #increment window index
                current_idx+= 1
                #save results periodically
                results_df= pd.DataFrame(recursive_preds).set_index("Date")
                #set save name with experiment and target metadata
                save_name= Path(f"Results/Data_experiments_qrf/{config['experiment_name']}_{target_name}_{h}m.csv")
                #ensure parent directory exists
                save_name.parent.mkdir(parents=True, exist_ok=True)
                #export results to csv
                results_df.to_csv(save_name)

#main execution entry point
if __name__ == "__main__":
    #init argument parser
    parser= argparse.ArgumentParser()
    #add config path argument
    parser.add_argument("--config", type=str, required=True)
    #parse arguments
    args= parser.parse_args()
    #load yaml config file
    with open(args.config, "r") as f:
        conf= yaml.safe_load(f)
    #run experiment logic
    run_experiment(conf)