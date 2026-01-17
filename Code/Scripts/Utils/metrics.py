import numpy as np

def calculate_crps(y_true, y_preds_quantiles, quantiles):
    """
    approximate CRPS via pinball loss fct (see thesis)
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
    return -calculate_crps(y, preds, q_grid) #negative value bc scikit-learn wants to maximize