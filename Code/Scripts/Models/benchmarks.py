import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import t
import warnings
from Utils.metrics import calculate_crps 



def ar_garch_model(y_series, max_ar=4):
    """fits AR(p)-GARCH(1,1) with Student-t errors (p based on AIC)
    """
    #initalize for aic to inf so first aic will be lower
    best_aic=np.inf
    #initialize best result
    best_res=None    
    #scale data by 100 as GARCH fail on very small numbers 
    y_scaled=y_series*100     
    #loop throrough AR lag p to find best Mean Equation
    for p in range(1, max_ar + 1):
        #supress warnings during fitting
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")                
            #def model
            model=arch_model(y_scaled, mean='AR', lags=p, vol='GARCH', p=1, q=1, dist='t')
            #estimate parameters
            res=model.fit(disp='off', show_warning=False)
        #if aic of this model is lower-> update
        if res.aic<best_aic:
            best_aic=res.aic
            best_res =res

            
    #if no AR model worked-> simpler const mean model
    if best_res is None:
        model= arch_model(y_scaled, mean='Constant', vol='GARCH', p=1, q=1, dist='t') #-> const mean
        best_res= model.fit(disp='off', show_warning=False)     #estimate parameters
    return best_res

def run_experiment(): 
    #namp experiment for output later 
    experiment_name="Benchmark_ARGARCH"
    #load data
    path='Code/Data/Cleaned_Data/data_stationary.csv'
    df=pd.read_csv(path, index_col='Date', parse_dates=True)
    #load yoy data for evaluation
    df_yoy= pd.read_csv('Code/Data/Cleaned_Data/data_yoy.csv', index_col='Date', parse_dates=True)#load yoy for evaluation
    #need to define stuff, normally defined in the config files
    targets=["Headline", "Core"]  
    retrain_step_months= 3       #re-estimate model every quarter 
    horizons= [3, 6, 9, 12]  #define all horizons
    eval_start_date= "2015-01-01" #start out of sample eval
    #define quantiles (for plotting vs crps calc)
    plot_qunat=[0.05, 0.16, 0.50, 0.84, 0.95]
    dense_quant= np.linspace(0.01, 0.99, 99)
    #get start date as timestamp
    eval_start_dt = pd.Timestamp(eval_start_date)
    #loop through targets and horizons to do recursive forecasts
    for target_name in targets:
        for h in horizons:
            #select cols for target
            if target_name=="Headline":
                target_col= f"target_headline_{h}m" #which horizon are forecasting
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
                #get end of current retrain window
                next_step_idx=min(current_idx +retrain_step_months, total_rows)
                test_indices= range(current_idx, next_step_idx) #indices to forecast in this window
                #break if no more test indices                
                if len(test_indices)==0:
                    break

                #train data: univariate-> only need target history up to current_idx
                y_full_series=df[target_col].iloc[:next_step_idx]       #full history up to end of window
                y_train= y_full_series.iloc[:current_idx].dropna()  #history up to start of window
                #need enough data points to fit model
                if len(y_train)< 30:
                    current_idx=next_step_idx
                    continue
                #fit model parameters (once per quarter)
                model_res=ar_garch_model(y_train)
                #check if model fitted/converged            
                if model_res is None:
                    current_idx=next_step_idx   #skip this window
                    continue
                #prediction for each month in the window
                for idx, date_idx in enumerate(test_indices):
                    #define forecast and target dates
                    forecast_date=df.index[date_idx]
                    target_date= forecast_date+pd.DateOffset(months=h)
                    
                    #get History specific to this date: garch needs immediate past res to predict var
                    current_history=y_full_series.iloc[:date_idx]
                    current_history_scaled =current_history * 100 #scale for garch
                    #forecast 1-step (=h-month) ahead, treat target as series
                    forecasts=model_res.forecast(params=model_res.params, start=len(current_history_scaled)-1, horizon=1, reindex=False)
                    #extract mean and var and descale
                    mu_pred=forecasts.mean.iloc[0, 0]/100
                    sigma_pred=np.sqrt(forecasts.variance.iloc[0, 0])/100
                    nu_est =model_res.params['nu'] #student-t degrees of freedom

                    #calc quantiles using t-dist
                    preds_plot=mu_pred+sigma_pred*t.ppf(plot_qunat, df=nu_est)
                    preds_dense= mu_pred+sigma_pred *t.ppf(dense_quant, df=nu_est)
                    #reconstruct to yoy changes: check if target date exists in yoy data
                    if target_date not in df_yoy.index:
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
                        mu_yoy=base_effect+(mu_pred*scaling_factor)
                        sigma_yoy=sigma_pred* scaling_factor    #scale only forecasted components volatility
                    #reconstruct quantiles to yoy
                    preds_plot_yoy= mu_yoy+sigma_yoy*t.ppf(plot_qunat, df=nu_est)
                    preds_dense_yoy= mu_yoy+sigma_yoy*t.ppf(dense_quant, df=nu_est)    

                    #evaluate CRPS
                    step_crps=calculate_crps([actual_val], preds_dense_yoy, dense_quant)
                    #store results
                    result = {'Date': forecast_date,'Target_date': target_date, 'Actual': actual_val, 'Forecast_median': preds_plot_yoy[2], 
                        'q05': preds_plot_yoy[0],'q16': preds_plot_yoy[1], 'q84': preds_plot_yoy[3],'q95': preds_plot_yoy[4], 
                        'Steps_CRPS': step_crps}
                    #append to recursive preds
                    recursive_preds.append(result)
                #to next window
                current_idx= next_step_idx

            #save results
            results_df = pd.DataFrame(recursive_preds)
            if not results_df.empty:
                results_df.set_index('Date', inplace=True)
                save_name = f"Results/Data_experiments/{experiment_name}_{target_name}_{h}m.csv"
                results_df.to_csv(save_name)
                print(f"Saved: {save_name}")

if __name__ == "__main__":
    run_experiment()

