import pandas as pd
import numpy as np
from quantile_forest import RandomForestQuantileRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import make_scorer
import sys
import os


#load Data
df = pd.read_csv('../../Data/Cleaned_Data/QRF_data.csv')
# Assuming 'target' is your inflation column and the rest are predictors
X = df.drop(columns=['target']) 
y = df['target']



#prepare data: time series split to avoid shuffling 
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
from metrics import crps_score


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