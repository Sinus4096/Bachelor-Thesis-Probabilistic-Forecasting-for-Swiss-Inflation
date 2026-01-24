from scipy.optimize import minimize
from scipy.stats import nct
#helper fct to fit skew-t parameters 

def fit_skew_t_params(target_quantiles, target_values):
    """
    Fits Skew-t parameters (df, nc, loc, scale) to match QRF quantiles.
    
    Args:
        target_quantiles: list of probabilities e.g. [0.05, 0.25, 0.5, 0.75, 0.95]
        target_values: the values predicted by QRF for those quantiles
    Returns:
        tuple: (df, nc, loc, scale)
    """
    # Initial guesses: df=10, nc(skew)=0, loc=median, scale=interquartile range
    initial_guess = [10, 0, np.median(target_values), np.std(target_values)]
    
    # Objective function: Minimize Sum of Squared Errors between QRF and Skew-t quantiles
    def objective(params):
        df, nc, loc, scale = params
        if df <= 2 or scale <= 0: # Constraints: df > 2 for variance, scale > 0
            return np.inf
        
        # Calculate theoretical quantiles for these parameters
        theoretical_values = nct.ppf(target_quantiles, df, nc, loc=loc, scale=scale)
        
        # Calculate Squared Error
        return np.sum((theoretical_values - target_values) ** 2)

    # Run optimization
    result = minimize(objective, initial_guess, method='Nelder-Mead')
    return result.x