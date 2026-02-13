import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from pathlib import Path


path ='Code/Data/Cleaned_Data/data_stationary_bvar.csv'
df_stationary=pd.read_csv(path, index_col='Date', parse_dates=True)
print(df_stationary.columns)

# 1. Prepare Features (X) 
# Use df_stationary but exclude all potential target columns
all_target_cols = [col for col in df_stationary.columns if 'target_' in col]
X = df_stationary.drop(columns=all_target_cols)

# Handle potential remaining NaNs (at the end of the df due to shifts)
X = X.ffill().dropna() 

# 2. Scale Features (Necessary for Lasso)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

# 3. Loop through targets to find best features
# We'll store how many times a feature is "selected" across horizons
selection_counts = pd.Series(0, index=X.columns)

targets_to_test = [col for col in df_stationary.columns if 'target_headline' in col]

from sklearn.linear_model import Lasso

# 1. Define a range of alphas from very small to large
alphas = np.logspace(-4, 1, 100)
feature_rankings = pd.Series(0, index=X.columns)

for target in targets_to_test:
    y = df_stationary[target].loc[X.index].dropna()
    curr_X = X_scaled.loc[y.index]
    
    # We use a simple Lasso loop to see which features survive at different alpha levels
    for a in alphas:
        model = Lasso(alpha=a).fit(curr_X, y)
        # Add 1 to the count for every feature that is NON-ZERO at this alpha level
        feature_rankings += (np.abs(model.coef_) > 1e-10).astype(int)

# The features with the highest counts are the most "robust" 
# (they stayed non-zero the longest as alpha increased)
top_5_robust = feature_rankings.sort_values(ascending=False).head(20)
print("Top 5 Robust Features (Lasso Path Ranking):")
print(top_5_robust)
top_7_features = [
    'Business_Confidence_EU', 
    'oilprices', 
    'infl_e_next_year', 
    'unemployment_rate', 
    'PPI', 
    'M3_change', 
    'Exchange_Rate_CHF', 
    #'Saron_Rate', 'kofbarometer', 'fin_spread', 'real_turnover', 'Manufacturing_EU', 'Wage_change', 'EU_fin_spread', 'M1_change',
    'headline_1m', 'core_1m','sin_cycle_1', 'cos_cycle_1',
       'sin_cycle_2', 'cos_cycle_2'
]
# 'Saron_Rate', 'kofbarometer', 'fin_spread', 'real_turnover', 'Manufacturing_EU', 'Wage_change', 'EU_fin_spread', 'M1_change',

# 2. Identify the target columns so we don't accidentally drop them
# This matches any column starting with 'target_'
target_cols = [col for col in df_stationary.columns if col.startswith('target_')]

# 3. Combine both lists
cols_to_keep = top_7_features + target_cols

# 4. Drop everything else
# This creates a new DF with ONLY the variables you want
df_small_bvar = df_stationary[cols_to_keep].copy()
CODE_DIR=Path(__file__).parent.parent
output_path=CODE_DIR /"Data"/"Cleaned_Data"
output_file4= output_path/'data_stationary_bvar.csv'
df_small_bvar.to_csv(output_file4, index=True)