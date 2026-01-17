import pandas as pd
import numpy as np
from quantile_forest import RandomForestQuantileRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error
import sys
import os
import yaml
import argparse
from pathlib import Path
from Code.Scripts.Utils.metrics import crps_score 

#use config files in order to run once Meinshausens default qrf and once a qrf with hyperparameter tuning
def load_config(config_path):
    """
    helper fct to load the config files
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)



#make function out of experiment to run the comparison experiment calling te config files:
def run_experiment(config):
    #want to know which one (default or tuning)
    print(f"run {config['experiment_name']}")
    #load df_stationary
    path='Code/Data/Cleaned_Data/data_stationary.csv'
    df=pd.read_csv(path, index_col='Date', parse_dates=True)

    #get target variables and forecast horizon from the config file
    targets=config['data']['targets']
    horizons=config['data']['horizons']
    #set recursive (out-of-sampe) prediction windos (->when stop training and update after how many months)
    eval_start_date= pd.Timestamp(config['data']['eval_start_date'])
    step_months=config['data']['retrain_step_months']

    #iterate through all targets 
    for target_name in targets:
        #iterate through all horizons defined in script 03
        for h in horizons:
            #setup data for this specific horizon of specific target (eg 3mont CPI forecast)
            if target_name =="Headline":
                target_col= f"target_headline_{h}m" #set target_col as defined in script 03
            else:
                target_col=f"target_core_{h}m"
            #make sure df_stationary contains the forecast horizon
            if target_col not in df.columns:
                continue
            #to follow the recursive testing, we don't drop rows where target_col is NAN but filter data available at that specific point in time:
            target_cols_to_drop= [col for col in df.columns if 'target_' in col]    #don't want target variable in X later
            recursive_preds = []    #initialize storage for out-of-sample predictions
            #start time loop at eval_start_date-> get index location of eval_start_date 
            start_idx = df.index.get_loc(eval_start_date)
            if isinstance(start_idx, slice):    #if get_loc returns slice->handle
                start_idx= start_idx.start
            total_rows= len(df) #get length of the original df

            #initialize the recursive loop and then iterate til to end of df
            current_idx= start_idx
            while current_idx <total_rows:
                #define the windows:
                next_step_idx= min(current_idx+step_months, total_rows) #define the next quarter 
                train_indices= range(0, current_idx)    #training set ranges from beginning up to the current index
                test_indices = range(current_idx, next_step_idx)    #testin from current index up until next quarter

                if len (test_indices)==0:
                    break #get out if reached the last row already

                #define ssubdf of original df up until next forecast step
                df_slice=df.iloc[:next_step_idx].copy()

                #separate X and Y
                X_slice= df_slice.drop(columns=target_cols_to_drop) #drop all cols starting with target_
                Y_slice= df_slice[target_col]
                #define train and test set
                X_train= X_slice.iloc[train_indices]
                Y_train = Y_slice.iloc[train_indices]
                X_test = X_slice.iloc[test_indices]
                Y_test =Y_slice.iloc[test_indices]

                #only drop NAN's for the training set: test might have NAN's in the end-> inference


        

#load Data
df= pd.read_csv('../../Data/Cleaned_Data/data_stationary.csv')
#split to X and Y

X= df.drop(columns=['target']) 
y= df['target']



#split data into train and test data (using 20% test data)
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]



#------------------------------------------------
#Meinshausen Default Model
#------------------------------------------------


# 2. Initialize the QRF model
# Meinshausen (2006) recommends min_samples_leaf=5 for QRF
default_qrf= RandomForestQuantileRegressor(
    n_estimators=1000,       # Meinshausen used 1000 for stability
    max_features='sqrt',     # Standard for high-dimensional inflation datasets
    min_samples_leaf=5,      # Meinshausen's suggested default
    random_state=42,
    n_jobs=-1                # Use all CPU cores
)

# 3. Fit the model
default_qrf.fit(X_train, y_train)

# 4. Expanded Quantiles for Density/Risk Assessment
# We use a wider range to capture the "tails" of inflation
quantiles = [0.05, 0.16, 0.50, 0.84, 0.95]
predictions = default_qrf.predict(X_test, quantile=quantiles)

# 5. Extracting Insights
df_preds = pd.DataFrame(predictions, columns=[f'q{int(q*100)}' for q in quantiles])
median_forecast = df_preds['q50']
inflation_uncertainty = df_preds['q84'] - df_preds['q16'] # Inter-quantile range
upside_risk = df_preds['q95'] - df_preds['q50']         # Distance to upper tail




#-----------------------------------------------------
#Optimized Model with Hyperparametertuning
#-----------------------------------------------------


#import metrics
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Code.Scripts.Utils.metrics import crps_score


qrf_to_tune = RandomForestQuantileRegressor(random_state=42)

param_dist = {
    'n_estimators': [500, 1000, 1500],
    'max_features': ['sqrt', 'log2', 0.3, 0.5],
    'min_samples_leaf': [2, 5, 10, 20], # Higher leaf size helps with small n=300
}

tuning_search = RandomizedSearchCV(
    estimator=qrf_to_tune,
    param_distributions=param_dist,
    scoring=crps_custom_scorer, 
    cv=tscv,
    n_iter=20, # Number of parameter combinations to try
    random_state=42,
    n_jobs=-1
)

print("Starting hyperparameter tuning for CRPS...")
tuning_search.fit(X_train, y_train)
best_qrf = tuning_search.best_estimator_

# 5. Evaluation & Comparison
q_grid = np.linspace(0.01, 0.99, 99)

# Predictions for Default
def_preds = default_qrf.predict(X_test, quantile=q_grid)
def_crps = crps_score(y_test, def_preds)

# Predictions for Tuned
tuned_preds = best_qrf.predict(X_test, quantile=q_grid)
tuned_crps = crps_score(y_test, tuned_preds)

print(f"\n--- Results ---")
print(f"Default QRF CRPS: {def_crps:.4f}")
print(f"Tuned QRF CRPS:   {tuned_crps:.4f}")
print(f"Best Params: {tuning_search.best_params_}")

# 6. Save for Thesis Plots (Fan Chart Data)
# Save the median and common quantiles for the Fan Chart
final_quantiles = [0.05, 0.16, 0.50, 0.84, 0.95]
results_df = pd.DataFrame(
    best_qrf.predict(X_test, quantile=final_quantiles),
    columns=['q05', 'q16', 'q50', 'q84', 'q95']
)
results_df['actual'] = y_test.values
results_df.to_csv('../../Results/qrf_forecasts.csv', index=False)