import numpy as np
from scipy.stats import nct
from scipy.integrate import quad

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
    return loss/len(quantiles)

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
