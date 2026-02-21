#ecb fit a skew-t distribution to all density forecasts to ensure comparability-> want to check if there is a distribution used by other references
#that fits better than the skew-t
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
#check whole thing on the actual evaluation targets: YoY inflation rates
path ='Code/Data/Cleaned_Data/data_yoy.csv'
df=pd.read_csv(path, index_col='Date', parse_dates=True)
#select only the two target variables in yoy not levels
variables=['Headline', 'Core']
#split the data based to avoid look-ahead bias
train_df= df[:'2012-07-01']

#define the distributions to compare
dist_names =['nct', 'skewnorm', 'norm', 'beta']
#intialize results storage
results_list=[]

#loup over variables and distributions
for var in variables:
    #remove NA's
    data= train_df[var].dropna().values    
    for name in dist_names:
        #get distribution object
        dist= getattr(stats, name)
        #fit dist to data
        params= dist.fit(data)            
        #calc goodness of fit metrics
        log_lik= np.sum(dist.logpdf(data, *params))  #log-likelihood
        d_stat, p_val= stats.kstest(data, name, args=params)  #Kolmogorov-Smirnov test
        #append results 
        results_list.append({'Variable': var, 'Distribution': name,'Log-Likelihood': log_lik, 'KS-Stat': d_stat,'P-Value': p_val})
        

#display results
results_df= pd.DataFrame(results_list)
for var in variables:
    print(f"\nBest Distributions for {var}:")
    print(results_df[results_df['Variable']== var].sort_values('Log-Likelihood', ascending=False))
    save_name=f"Code/Scripts/Plots_and_Tables/05_diagnostic_distribution_analysis/dist_fits_{var}.csv"
    results_df.to_csv(save_name) 

#-> skew-t fits best on training data (for Core 1. and for headline 2. best)
#why not Johnsonsu?skew-t is often more robust in economic forecasting than the Johnson SU.

