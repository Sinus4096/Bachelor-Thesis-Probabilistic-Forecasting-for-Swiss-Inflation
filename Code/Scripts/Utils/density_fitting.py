import numpy as np
from scipy.stats import nct
from scipy.optimize import least_squares
from scipy.optimize import least_squares
import pymetalog as pm

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



def fit_metalog(quantiles, probs, term_limit=5):
    """
    Fits Metalog to QRF quantiles. 
    Converts data to lists to avoid pymetalog's NumPy compatibility bugs.
    """
    # 1. Ensure quantiles are unique (Metalog requirement)
    # We add a tiny bit of noise if they aren't, to keep the full distribution
    if len(np.unique(quantiles)) < len(quantiles):
        quantiles = quantiles + np.linspace(0, 1e-9, len(quantiles))
    
    # 2. CONVERT TO LISTS
    # This bypasses the 'ValueError: truth value of an array...' inside the library
    x_list = quantiles.tolist() if isinstance(quantiles, np.ndarray) else list(quantiles)
    p_list = probs.tolist() if isinstance(probs, np.ndarray) else list(probs)
    
    # 3. Fit
    m = pm.metalog(x=x_list, probs=p_list, term_limit=term_limit, boundedness='u')
    return m
