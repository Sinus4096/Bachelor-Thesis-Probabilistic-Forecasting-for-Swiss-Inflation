import sys
import numpy as np
import pandas as pd
from arch import arch_model
from pyparsing import Path
from scipy.stats import nct
import warnings
#get path for utils
current_dir=Path(__file__).resolve().parent
#get project root
scripts_root = current_dir.parent.parent   
sys.path.insert(0, str(scripts_root))
from Scripts.Utils.density_fitting import fit_skew_t
from Scripts.Utils.metrics import calculate_crps, calculate_crps_quantile, calculate_rmse 




def ar_garch_model(y_series, max_ar=4):
    """fits AR(p)-GARCH(1,1) with Student-t errors (p based on AIC)
    """
    #initalize for aic: fit const mean firs to set baseline
    base_model=arch_model(y_series, mean='Constant', vol='GARCH', p=1, q=1, dist='skewt')
    #get results of base model
    best_res= base_model.fit(disp='off', show_warning=False)
    best_aic= best_res.aic #initialize best aic
    #loop throrough AR lag p to find best Mean Equation
    for p in range(1, max_ar + 1):
        #supress warnings during fitting
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")                
            #fit model
            model=arch_model(y_series, mean='AR', lags=p, vol='GARCH', p=1, q=1, dist='skewt')
            #estimate parameters
            res=model.fit(disp='off', show_warning=False)
        #if aic of this model is lower-> update
        if res.aic<best_aic:
            best_aic=res.aic
            best_res =res

            
    #if no AR model worked/increased AIC too much-> simpler const mean model
    if best_res is None:
        model= arch_model(y_series, mean='Constant', vol='GARCH', p=1, q=1, dist='skewt') #-> const mean
        best_res= model.fit(disp='off', show_warning=False)     #estimate parameters
    return best_res

def run_experiment(): 
    #namp experiment for output later 
    experiment_name="Benchmark_ARGARCH"
    #load data
    project_root=current_dir.parent.parent
    #get path to selected data
    data_path =project_root /"Data"/ "Cleaned_Data"/"data_stationary.csv"
    df =pd.read_csv(data_path, index_col='Date', parse_dates=True)
    #load yoy data for evaluation
    df_yoy= pd.read_csv(project_root/"Data"/"Cleaned_Data"/"data_yoy.csv", index_col='Date', parse_dates=True)#load yoy for evaluation
    #need to define stuff, normally defined in the config files
    targets=["Headline", "Core"]  
    retrain_step_months= 3       #re-estimate model every quarter 
    horizons= [3, 6, 9, 12]  #define all horizons
    eval_start_date= "2013-07-01" #start out of sample eval
    #snb forecasts once per quarter: in march, june, september, december
    snb_months=[3, 6, 9, 12]
    #define quantiles (for plotting vs crps calc)
    plot_qunat=[0.05, 0.16, 0.50, 0.84, 0.95]
    dense_quant= np.linspace(0.01, 0.99, 99)
    #get start date as timestamp
    eval_start_dt = pd.Timestamp(eval_start_date)

    #use rolling window to capture structural breaks (set to minimum bc only 25 y data and dont want post covid era to depend on financial cirsis and peg era)
    rolling_window_size=10*12 
    #loop through targets and horizons to do recursive forecasts
    for target_name in targets:
        for h in horizons:
            #select cols for target
            if target_name=="Headline":
                target_col= f"target_headline_{h}m"      #which horizon are forecasting
                yoy_col= "Headline" #for evaluation (need to deannualize)
                yoy_raw= "Headline_level"   #want to evaluate on yoy changes
            else:   #same if core
                target_col=f"target_core_{h}m"
                yoy_col= "Core"
                yoy_raw="Core_level"
            #check if target exists in data
            if target_col not in df.columns:
                continue
            #initilalize storage for results
            recursive_preds=[]            
            #numerical index in the dataframe where evaluation starts
            start_idx=df.index.get_loc(eval_start_dt)
            if isinstance(start_idx, slice): # in case of multiple matches take first
                start_idx=start_idx.start
            #get total rows
            total_rows=len(df)
            current_idx =start_idx  #initialize current index for recursive loop

            #recursive forecasting
            while current_idx<total_rows:
                #identify dates now ->when forecast happens
                forecast_date=df.index[current_idx]
                #check whether is an SNB forecast month
                if forecast_date.month not in snb_months:
                    current_idx+=1   #move to next month
                    continue
                target_date= forecast_date+pd.DateOffset(months=h)  #target date
                #train data: univariate-> only need target history up to current_idx: need most recent monthly data
                y_full_series= df[target_col].iloc[:current_idx -(h-1)]      #full history up to end of window (only know j-month change that was completed before the change)
                #to define train data need to cut off old data if needed
                if len(y_full_series)> rolling_window_size:
                    y_train= y_full_series.iloc[-rolling_window_size:].dropna()  
                else:
                    y_train= y_full_series.dropna()
                #need enough data points to fit model
                if len(y_train)< 30:
                    current_idx+=retrain_step_months   #skip to next window
                    continue
                #fit model parameters (once per quarter)
                model_res=ar_garch_model(y_train)
                #check if model fitted/converged            
                if model_res is None:
                    current_idx+=retrain_step_months   #skip this window
                    continue

                #forecast1 step ahead (=h-month ahead)
                forecasts=model_res.forecast(horizon=1, reindex=False)
                #extract mean and var
                mu_pred=forecasts.mean.iloc[0, 0]
                sigma_pred=np.sqrt(forecasts.variance.iloc[0, 0])
                #extract skew-t parameters from GARCH fit
                dist_params= model_res.params.iloc[-2:]
                

                #reconstruct to yoy changes: check if target date exists in yoy data
                if target_date not in df_yoy.index:
                    current_idx+=retrain_step_months   #skip this window
                    continue 
                actual_val=df_yoy.loc[target_date, yoy_col] #get actual yoy value
                #for 12 m forecast no adjustment needed-> mu and sigma are already yoy changes
                if h==12:
                    mu_yoy=mu_pred
                    sigma_yoy= sigma_pred
                else:
                    #get history date for base effect
                    months_back=12-h    #months to go back for base effect
                    history_date =forecast_date-pd.DateOffset(months=months_back)
                    #get level prices
                    p_t=np.log(df_yoy.loc[forecast_date, yoy_raw])
                    p_hist =np.log(df_yoy.loc[history_date, yoy_raw])
                        
                    #compute base effect
                    base_effect=(p_t-p_hist)*100                        
                    #de-annualize parameters for mean and volatility
                    scaling_factor=h/12
                    mu_yoy=base_effect+(mu_pred*scaling_factor) #reconstruct mean
                    sigma_yoy=sigma_pred* np.sqrt(scaling_factor)    #scale only forecasted components volatility
                #ask model how many params needed
                n_shape_params= model_res.model.distribution.num_params
                #extract correct dist params
                dist_params= model_res.params.iloc[-n_shape_params:]
                #reconstrunct quantiles to yoy
                preds_plot_yoy = mu_yoy + sigma_yoy * model_res.model.distribution.ppf(plot_qunat, dist_params)
                preds_dense_yoy = mu_yoy + sigma_yoy * model_res.model.distribution.ppf(dense_quant, dist_params)
                               
                #get rmse of meadian forecast
                sq_error = calculate_rmse(actual_val, preds_plot_yoy[2]) 
                #use skew t to fit data for parametric crps eval
                y_fit_data=preds_dense_yoy.flatten() #data to fit
                skew_params=fit_skew_t(y_fit_data, dense_quant)
                #calc pit values for crps eval
                pit_val=nct.cdf(actual_val, skew_params[0], skew_params[1], loc=skew_params[2], scale=skew_params[3])
                #calc parametric crps based on fitted skew-t
                param_crps= calculate_crps(actual_val, skew_params)
                #empirical crps to see what fitting cost us
                empirical_crps= calculate_crps_quantile([actual_val], preds_dense_yoy.reshape(1,-1), dense_quant)
                
                #store results
                result= {'Date': forecast_date, 'Target_date': target_date, 'Actual': actual_val, 'Forecast_median': preds_plot_yoy[2], 
                    'q05': preds_plot_yoy[0], 'q16': preds_plot_yoy[1], 'q84': preds_plot_yoy[3], 'q95': preds_plot_yoy[4], 'Squared_Error': sq_error,
                    'Empirical_CRPS': empirical_crps, 'Parametric_CRPS': param_crps, 'PIT': pit_val, 'df_skewt': skew_params[0], 'nc_skewt': skew_params[1], 
                    'loc_skewt': skew_params[2], 'scale_skewt': skew_params[3]}
                #append to recursive preds
                recursive_preds.append(result)
                #to next window
                current_idx+=retrain_step_months

            #save results
            results_df = pd.DataFrame(recursive_preds)
            if not results_df.empty:
                results_df.set_index('Date', inplace=True)
                save_name = f"Results/Data_experiments_benchmark/{experiment_name}_{target_name}_{h}m.csv"
                results_df.to_csv(save_name)

if __name__ == "__main__":
    run_experiment()

