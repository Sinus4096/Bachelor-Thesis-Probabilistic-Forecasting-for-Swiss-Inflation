import numpy as np
from scipy.stats import nct
from scipy.optimize import least_squares
from scipy.stats import skewnorm
from scipy.optimize import least_squares

#helper fct to fit skew-normal parameters -> can compare the models
#see script 04_diagnostic_distribution_analysis.py for why not fitting other distributions and comparison

def fit_skew_t(quantiles_yoy, quantile_levels):
    """fct only needed for qrf, as BVAR uses its own density fitting approach"""
    #define error fct for  optimization: want to min diff between qrf quantiles and theoretical quantiles of skew-t distribution (as function for simplicity)
    def loss_fct(params, x_quantiles, q_levels):
        df, nc, loc, scale=params   #unpack params
        #have constraints that df>2, scale>0
        if df<=2 or scale<=0:
            return 1e6  #large penalty
        #get theoretical quantiles 
        theo_quantiles=nct.ppf(q_levels, df, nc, loc=loc, scale=scale)
        #return difference between theo and empirical quantiles 
        return theo_quantiles-x_quantiles
    #initial parameter guesses
    init_params=[10.0, 0.0, np.median(quantiles_yoy), np.std(quantiles_yoy)]
    #optimize the loss fct 
    bounds = ([1.0, -10.0, -np.inf, 0.001], [100.0, 10.0, np.inf, np.inf])   #set bounds:df >= 1, scale >= 0.001
    res= least_squares(loss_fct, init_params, args=(quantiles_yoy, quantile_levels), bounds=bounds)
    return res.x


def fit_skew_normal(quantiles_yoy, quantile_levels):
    """
    Fits a Skew-Normal distribution to the provided quantiles.
    """
    
    # Define error fct: minimize diff between empirical and theoretical quantiles
    def loss_fct(params, x_quantiles, q_levels):
        a, loc, scale = params  # unpack params (no 'df' anymore)
        
        # Constraint: scale > 0
        if scale <= 0:
            return 1e6  # large penalty
            
        # Get theoretical quantiles using skewnorm
        # 'a' is the shape parameter driving skewness
        theo_quantiles = skewnorm.ppf(q_levels, a, loc=loc, scale=scale)
        
        # Return difference
        return theo_quantiles - x_quantiles

    # Initial parameter guesses: [shape(skew), loc, scale]
    # We guess 0 skewness (normal), median for loc, and std for scale
    init_params = [0.0, np.median(quantiles_yoy), np.std(quantiles_yoy)]
    
    # Set bounds: 
    # Shape (a): [-inf, inf] (or restrict to +/- 10 if you want to prevent extreme skew)
    # Loc: [-inf, inf]
    # Scale: [0.001, inf]
    bounds = ([-np.inf, -np.inf, 0.001], [np.inf, np.inf, np.inf])
    
    # Optimize
    res = least_squares(loss_fct, init_params, args=(quantiles_yoy, quantile_levels), bounds=bounds)
    
    return res.x

