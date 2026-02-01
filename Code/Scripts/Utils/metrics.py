import numpy as np
from scipy.stats import nct
from scipy.integrate import quad
import pymetalog as pm

#approximation of crps for hyperparameter tuning in qrf-> evaluation of quantile predictions
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

def qrf_crps_scorer(estimator, X, y):
    """
    custom scorer hyperparameter tuning"""
    #define grid
    q_grid=np.linspace(0.01, 0.99, 99) 
    #generate quantile preds using qrf's weighted empirical cdf inversion
    preds=estimator.predict(X, quantile=q_grid)
    return -calculate_crps_quantile(y, preds, q_grid) #negative value bc scikit-learn wants to maximize



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




def calculate_crps_metalog(y_true, metalog_model, quantiles, term=5):
    """
    Calculates CRPS with error handling for pymetalog's numerical instability.
    """
    from scipy.integrate import quad
    import numpy as np
    import pymetalog as pm

    # Use the predicted quantiles to define a tighter integration range
    # Metalog is unstable in far tails; 20% buffer is safer than 50%
    q_min, q_max = np.min(quantiles), np.max(quantiles)
    spread = q_max - q_min
    lower_lim = q_min - (spread * 0.2)
    upper_lim = q_max + (spread * 0.2)
    median_val = np.median(quantiles)

    def integrand(x):
        try:
            # Attempt to get CDF from the library
            res = pm.pmetalog(metalog_model, q=[float(x)], term=term)
            
            # If library returns empty list or NaN, fallback to step function
            if res is None or len(res) == 0 or np.isnan(res[0]):
                cdf_val = 0.0 if x < median_val else 1.0
            else:
                cdf_val = np.clip(float(res[0]), 0, 1)
        except:
            # Emergency fallback for numerical crashes
            cdf_val = 0.0 if x < median_val else 1.0
            
        heaviside = 1.0 if x >= y_true else 0.0
        return (cdf_val - heaviside)**2

    # Numerical integration with a tighter tolerance to speed up
    crps_val, _ = quad(integrand, lower_lim, upper_lim, limit=50, epsabs=1e-4)
    return crps_val

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