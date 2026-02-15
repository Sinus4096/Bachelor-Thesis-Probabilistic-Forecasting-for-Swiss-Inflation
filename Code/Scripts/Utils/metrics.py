import warnings
import numpy as np
from scipy.stats import nct
from scipy.integrate import quad
import shap
import pandas as pd

#evaluation of quantile predictions
#-----------------------------
def calculate_crps_quantile(y_true, y_preds_quantiles, quantiles):
    """
    approximate CRPS via pinball loss fct 
    """
    #ensure actual y value is col vec
    y_true=np.array(y_true).reshape(-1, 1)
    #intialize loss
    loss= 0.0
    #iterate through Q_alpha in provided grid
    for idx, qunat in enumerate(quantiles):
        #calculate residuals
        errors =y_true-y_preds_quantiles[:, idx:idx+1]
        #add caltc. pinball loss to total loss
        loss +=np.mean(np.maximum(qunat* errors, (qunat- 1)*errors))
    
    #devide by nr of quantiles -> approx integral across distrib.
    return 2.0*(loss/len(quantiles))




#2.crps for all models to compare and evaluate their  distribution
#-----------------------------
def calculate_crps(y_true, params):
    """ cacl crps for a skew-t distribution against actual value"""
    #unpack params
    df, nc, loc, scale=params
    #handle NAN if fitting faile
    if np.isnan(df):
        return np.nan
    #set integration limits: go far out into the tails to ensure capture fat tails
    lower_lim=nct.ppf(1e-6, df, nc, loc=loc, scale=scale)
    upper_lim=nct.ppf(1 - 1e-6, df, nc, loc=loc, scale=scale)
    #quickly ensure that bounds cover y_true
    lower_lim= min(lower_lim, y_true-10*scale)
    upper_lim= max(upper_lim, y_true+10*scale)
    def integrand_left(z):
        """integral from lower_bound to y_true"""
        return nct.cdf(z, df, nc, loc=loc, scale=scale)**2
    def integrand_right(z):
        """integral from y_true to upper_bound"""
        return (1.0 -nct.cdf(z, df, nc, loc=loc, scale=scale))**2
    
    #perform integration-> calc crps
    res_left,_= quad(integrand_left, lower_lim, y_true)
    res_right,_= quad(integrand_right, y_true, upper_lim)
    return res_left+res_right





#3.RMSE to compare their point forecast accuracy
#-----------------------------
def calculate_rmse(actual, predicted_median):
    """
    calc squared error of single observation, for mean part-> aggregate results later"""
    Sq_error=(actual- predicted_median)**2
    return Sq_error



#shapley value
#---------------------------------
def shap_values(model_obj, X_input, X_train=None, model_type='linear', linear_coeffs=None, linear_const=0.0):
    """calculate Shapley values need to differentiate between linear and non linear models
    ->linear models: Exact Calculation (Coefficients*Values) because it is faster and mathematically precise
     ->QRF: shap (=an approximation) """
    #initialize dict to store contributions
    shap_contributions={} 
    #Tree-Based Models (QRF)
    #-------------------------------------------------------
    #check if model type is tree based
    if model_type == 'tree':     
        #define inner function to handle data conversion to dictionary
        def extract_to_dict(vals, expected, feature_names): 
            #extract first element if values are in a list
            if isinstance(vals, list):vals=vals[0] 
            #handle multi-dimensional arrays by taking the first row
            if len(vals.shape) >1: vals =vals[0] 
            #extract first element of expected value if array
            if isinstance(expected, np.ndarray): expected = expected[0] 
            
            #loop through feature names and assign values to dictionary
            for i, col in enumerate(feature_names): 
                #map feature name to its corresponding shap value
                shap_contributions[f'Shap_{col}'] =vals[i] 
            #store the expected value as the shap constant
            shap_contributions['Shap_Constant']=expected 

        #begin attempt to use fast explainer
        try: 
            #Attempt 1: Fast TreeExplainer
            #initialize the tree explainer object
            explainer=shap.TreeExplainer(model_obj) 
            #calculate shap values without checking additivity for speed
            shap_vals= explainer.shap_values(X_input, check_additivity=False) 
            #call helper to format and store results
            extract_to_dict(shap_vals, explainer.expected_value, X_input.columns) 
            
        #handle cases where tree explainer fails
        except Exception: 
            #Attempt 2: Fallback to KernelExplainer (for quantile-forest)
            #check if training data is provided for kernel estimation
            if X_train is None: 
                #raise error if background data is missing
                raise ValueError("X_train is required for QRF/KernelExplainer fallback") 

            #capture feature names to fix warning
            feature_names=X_train.columns.tolist() 

            #wrapper to add column names back and predict median
            def predict_wrapper(data): 
                #check if data is numpy array to convert back to dataframe
                if isinstance(data, np.ndarray): 
                    #re-apply feature names for model consistency
                    data = pd.DataFrame(data, columns=feature_names) 
                #predict the 0.5 quantile and flatten output
                return model_obj.predict(data, quantiles=[0.5]).flatten() 

            #suppress warnings during kmeans and shap calculation
            with warnings.catch_warnings(): 
                #ignore simple warnings for cleaner output
                warnings.simplefilter("ignore") 
                #summarize background data using 10 centroids
                background=shap.kmeans(X_train, 10)  
                #initialize the kernel explainer with wrapper
                explainer =shap.KernelExplainer(predict_wrapper, background) 
                #calculate shap values in silent mode
                shap_vals =explainer.shap_values(X_input, silent=True) 
            
            #call helper to format and store kernel results
            extract_to_dict(shap_vals, explainer.expected_value, X_input.columns) 

    #Linear Models (AR-GARCH, BVAR)
    #---------------------------------------------------
    #check if model type is linear
    elif model_type =='linear': 
        #loop through coefficients and multiply by input values
        for feature_name, coef_val in linear_coeffs.items(): 
            #X_input is a Series with matching index (e.g., BVAR with named lags)
            if feature_name in X_input.index: 
                #find matching value in input data
                val =X_input[feature_name]     
                #calc shap value as Coefficients*Values
                shap_contributions[f'Shap_{feature_name}'] =coef_val*val   
            #AR-GARCH style (arch package uses "y[1]", "y[2]")
            elif 'y[' in feature_name: 
                #parse the lag index from the string name
                lag_idx=int(feature_name.split('[')[1].split(']')[0]) 
                #X_input is sorted chronologically, y[1] is the last item
                val=X_input.iloc[-lag_idx]  
                #calc shap value as Coefficients*Values
                shap_contributions[f'Shap_Lag_{lag_idx}']= coef_val*val   
        #add constant term contribution            
        shap_contributions['Shap_Constant'] =linear_const 
    #return the final dictionary of contributions
    return shap_contributions
