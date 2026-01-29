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
    #set integration limits (=mean +/- 10 std devs)
    lower_lim=loc-10*scale
    upper_lim=loc+10*scale
    
    #perform integration-> calc crps
    crps_val, _=quad(lambda z: (nct.cdf(z, df, nc, loc=loc, scale=scale)-(1.0 if z >=y_true else 0.0))**2,lower_lim, upper_lim)
    return crps_val
