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
train_df= df[:'2015-12-31']
test_df= df['2016-01-01':]
#define the distributions to compare
dist_names =['nct', 'skewnorm', 'johnsonsu', 'norm']
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
    print(results_df[results_df['Variable'] == var].sort_values('Log-Likelihood', ascending=False))

#-> skew normal and johnsonsu fit better than skew-t for both variables: will take skew normal for simplicity

#visualize with qq-plots
#----------------------------
#prepare data for easy plotting
headline_data= train_df['Headline'].dropna().values
core_data= train_df['Core'].dropna().values
#redefine distributions with nicer names for plotting
dists= [('skewnorm', 'Skew-Normal'), ('johnsonsu', 'Johnson SU'), ('nct', 'Skew-T'), ('norm', 'Normal')]
#create subplots
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
# 1. Global Styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.titleweight'] = 'bold'

# Define a professional color palette
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
#loop through variables and distributions to create qq-plots
for row_idx, (data, var_name) in enumerate([(core_data, 'Core Inflation'), (headline_data, 'Headline Inflation')]):
    for col_idx, (dist_name, display_name) in enumerate(dists):
        ax=axes[row_idx, col_idx] #select subplot
        #fit and plot
        dist= getattr(stats, dist_name)    #get distribution object
        params= dist.fit(data)   #fit dist to data
        
        #get QQ-plot data
        (osm, osr), (slope, intercept, r)= stats.probplot(data, dist=dist_name, sparams=params)
        #data points define
        ax.scatter(osm, osr, alpha=0.6, s=40, color=colors[col_idx], edgecolor='w', linewidth=0.5, label=f'R² = {r**2:.3f}')
        
        #reference line
        ax.plot(osm, slope*osm +intercept, color='black', linestyle='--', linewidth=1.5, alpha=0.8)
        #labeling
        ax.set_title(f"{var_name}\n{display_name}", fontsize=14, pad=15)
        ax.legend(loc='lower right', frameon=True, fontsize=10)
        #remove top/right spines
        sns.despine(ax=ax)
        #only show labels on the edges to reduce clutter
        if col_idx== 0:
            ax.set_ylabel("Observed Quantiles (%)", fontsize=12, fontweight='bold')
        else:
            ax.set_ylabel("") 
        if row_idx== 1:
            ax.set_xlabel("Theoretical Quantiles", fontsize=12, fontweight='bold')
        else:
            ax.set_xlabel("")

plt.suptitle("Probability Distribution Fit Analysis: Swiss CPI", fontsize=22, y=1.02, fontweight='bold')
plt.tight_layout()
plt.show()