import warnings
import numpy as np
from scipy.stats import nct
from scipy.integrate import quad
import shap
import pandas as pd
from sklearn.base import clone
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



#4.CRPS via quantile preds for empirical crps calculation: want to see how much smoothing the skew-t fit changed the result
#-----------------------------
def calculate_empirical_crps(y_true, y_preds, quantiles):
    """
    calculates CRPS approximation via pinball loss from quantile predictions"""
    #ensure inputs are numpy arrays
    y_preds=np.array(y_preds).flatten()
    quantiles= np.array(quantiles)
    
    #pinball loss calculation as in calculate_crps_quantile funtion
    diff=y_true -y_preds
    loss= np.where(diff>= 0, diff*quantiles, -diff *(1-quantiles))
    return 2*np.mean(loss)






def shap_values(model_obj, X_input, X_train=None, model_type='linear', linear_coeffs=None, linear_const=0.0):
    """ calculate Shapley values need to differentiate between linear and non linear models
    -> linear models: Exact Calculation (Coefficients*Values) because it is faster and mathematically precise
     ->QRF: shap (=an approximation) """
    #initialize dict to store contributions
    shap_contributions = {}

    #Tree-Based Models (QRF)
    #-------------------------------------------------------
    if model_type == 'tree':    
        # 1. Try the much faster TreeExplainer first
        def extract_to_dict(vals, expected, feature_names):
            if isinstance(vals, list): vals = vals[0]
            if len(vals.shape) > 1: vals = vals[0]
            if isinstance(expected, np.ndarray): expected = expected[0]
            
            for i, col in enumerate(feature_names):
                shap_contributions[f'Shap_{col}'] = vals[i]
            shap_contributions['Shap_Constant'] = expected

        try:
            # Attempt 1: Fast TreeExplainer
            explainer = shap.TreeExplainer(model_obj)
            shap_vals = explainer.shap_values(X_input, check_additivity=False)
            extract_to_dict(shap_vals, explainer.expected_value, X_input.columns)
            
        except Exception:
            # Attempt 2: Fallback to KernelExplainer (for quantile-forest)
            if X_train is None:
                raise ValueError("X_train is required for QRF/KernelExplainer fallback")

            # Capture feature names to fix warning
            feature_names = X_train.columns.tolist()

            # Wrapper to add column names back and predict median
            def predict_wrapper(data):
                if isinstance(data, np.ndarray):
                    data = pd.DataFrame(data, columns=feature_names)
                return model_obj.predict(data, quantiles=[0.5]).flatten()

            # Suppress warnings during kmeans and shap calculation
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                background = shap.kmeans(X_train, 10) 
                explainer = shap.KernelExplainer(predict_wrapper, background)
                shap_vals = explainer.shap_values(X_input, silent=True)
            
            extract_to_dict(shap_vals, explainer.expected_value, X_input.columns)

    #Linear Models (AR-GARCH, BVAR)
    #---------------------------------------------------
    elif model_type =='linear':
        #loop through coefficients and multiply by input values
        for feature_name, coef_val in linear_coeffs.items():
            #X_input is a Series with matching index (e.g., BVAR with named lags)
            if feature_name in X_input.index:
                val =X_input[feature_name]    #find mathcing value in input data
                shap_contributions[f'Shap_{feature_name}'] =coef_val*val  #calc shap value as Coefficients*Values
            #AR-GARCH style (arch package uses "y[1]", "y[2]")
            elif 'y[' in feature_name:
                lag_idx=int(feature_name.split('[')[1].split(']')[0])
                # X_input is sorted chronologically, y[1] is the last item
                val=X_input.iloc[-lag_idx] 
                shap_contributions[f'Shap_Lag_{lag_idx}']= coef_val*val  #calc shap value as Coefficients*Values
        #add constant term contribution           
        shap_contributions['Shap_Constant'] =linear_const
    return shap_contributions
