import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import matplotlib.ticker as mtick
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
import statsmodels.api as sm

#set plotting style 
sns.set_theme(style="white", palette="muted")
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

#get df
path ='Code/Data/Cleaned_Data/data_stationary.csv'
df=pd.read_csv(path, index_col='Date', parse_dates=True)
#differ between targets and predictors:
target_cols=[col for col in df.columns if 'target_' in col]
X= df.drop(columns=target_cols)
#only look at train data during analysis
X=X[:'2012-07-01']

#Correlation matrix
#----------------------------
#correlation function
corr_matrix= X.corr()
#initialize plot:
plt.figure(figsize=(14, 10))
mask= np.triu(np.ones_like(corr_matrix, dtype=bool))     #hide redundant upper half
cmap= sns.diverging_palette(230, 20, as_cmap=True)   #professional Red-Blue palette
#plot heatmap for corr
sns.heatmap(corr_matrix, mask=mask, cmap=cmap, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .8})
plt.title("Correlation Structure of Predictors (Training Set)", fontsize=16, pad=20, weight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
#plt.savefig('Correlation_Heatmap.png', dpi=300)
#observation:
#The matrix shows rectangular blocks of red and blue. This indicates that variables like M1, M2, and the Saron Rate were 
# moving in near-perfect lockstep during 2001–2012. This provides a definitive mandate for the Stock and Watson (2002) approach.
#It proves that the 33 variables are actually just multiple shadows cast by few primary economic engines.

#VIF values
#----------------------
#is numerical metric that measures multicollinearity->how much one predictor is a linear combination of others.
#add constant
X_vif= sm.add_constant(X)
#initialize df
vif_data = pd.DataFrame()
#get cols from original df
vif_data["Variable"]= X_vif.columns
#calc vif values
vif_data["VIF"]= [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]

print(vif_data)

#into bar chart:
vif_sorted= vif_data.sort_values('VIF', ascending=True)
plt.figure(figsize=(10, 8))
plt.barh(vif_sorted['Variable'], vif_sorted['VIF'], color='darkred', alpha=0.8)
plt.axvline(x=10, color='grey', linestyle='--', label='Threshold (10)')
plt.xscale('log')
plt.xlabel('Variance Inflation Factor (Log Scale)', weight='bold')
plt.title('Predictor Redundancy Analysis', fontsize=16, weight='bold')
plt.legend()
plt.tight_layout()
#plt.savefig('VIF_Analysis_Chart.png', dpi=300)


#Eigenvalue analysis
#---------------------------
#background: each eigenvalue represents the amount of variance captured by a specific factor. An eigenvalue > 1 means that a 
#single factor captures more information than one individual raw variable.
#for analysis need standardized features
scaler= StandardScaler()
X_std= scaler.fit_transform(X)
#get cov matrix of the standardized features
cov_matrix= np.cov(X_std.T)
#get eigenvalues of the cov matrix
eigenvalues, _= np.linalg.eig(cov_matrix)
eigenvalues_sorted = np.sort(eigenvalues)[::-1]

print(eigenvalues_sorted)
#observation:
#The first eigenvalue (7.48) is significantly higher than the rest. This means the primary factor is incredibly dominant in the 
# training period. This supports the use of Principal Component Analysis (PCA) to summarize the predictors. Using the first 
#few components captures the vast majority of the predictable dynamics.

#visual with scree plot

#plot a scree plot: to visualize the amount of variation in a dataset explained by each principal component
#calc proportion of total variance captured by each conponent
var_exp= eigenvalues_sorted/ np.sum(eigenvalues_sorted)
#total of variance explained as we add more components
cum_var_exp= np.cumsum(var_exp)
#initialize plot
fig, ax1=plt.subplots(figsize=(12, 7))
#plot eigenvalues
ax1.set_xlabel('Principal Component Number', fontsize=12)
ax1.set_ylabel('Eigenvalue (Variance)', color='tab:blue', fontsize=12)
ax1.plot(range(1, len(eigenvalues_sorted) + 1), eigenvalues_sorted, marker='o', color='tab:blue', linewidth=2, label='Individual Eigenvalue')
ax1.axhline(y=1, color='red', linestyle='--', alpha=0.6, label='Kaiser Criterion (λ=1)')
ax1.grid(alpha=0.3)

#plot cumulative variance on second axis
ax2 = ax1.twinx()
ax2.set_ylabel('Cumulative Variance Explained', color='tab:orange', fontsize=12)
ax2.plot(range(1, len(eigenvalues_sorted) + 1), cum_var_exp, marker='s', color='tab:orange', linestyle=':', alpha=0.7, label='Cumulative Variance')
ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

plt.title("Scree Plot & Cumulative Information Capture", fontsize=16, weight='bold', pad=15)
fig.legend(loc='upper right', bbox_to_anchor=(0.85, 0.85))
plt.tight_layout()
#plt.savefig('Scree_Plot.png', dpi=300)
#observation:
#Individual Eigenvalues (Blue Line): This line represents the variance captured by each individual principal component. The sharp ellbow
# at the fourth or fifth component indicates that the vast majority of the predictable dynamics in the 33-variable dataset are 
# concentrated in just a few latent factors.
#Kaiser Criterion (Red Line): Following standard econometric practice, the horizontal line at $\lambda=1$ represents the 
#threshold where a single principal component explains as much variance as one original standardized variable. In this dataset, 
# approximately 10–11 components satisfy this criterion, though the most significant information is captured by the first five.
#Cumulative Variance (Orange Line): This secondary axis quantifies the total information capture. As shown, the first five 
#factors alone capture approximately 65–70% of the total variation in the 33 macroeconomic predictors. By the 15th component, 
# nearly 90% of the information set is accounted for.

#-> this analysis justifies the use of PCA to shrink our number of features

