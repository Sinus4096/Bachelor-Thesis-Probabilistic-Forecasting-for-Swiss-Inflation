import pandas as pd
from quantile_forest import RandomForestQuantileRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import make_scorer
from utils.metrics import crps_score

# --- 1. Custom Scorer for CRPS ---
def crps_scorer_func(estimator, X, y):
    # Predict a dense grid for CRPS calculation
    q_grid = np.linspace(0.01, 0.99, 99)
    preds = estimator.predict(X, quantile=q_grid)
    return -crps_score(y, preds) # Negative because CV maximizes

crps_scorer = make_scorer(crps_scorer_func, greater_is_better=False)

# --- 2. Setup Tuning ---
# Using small min_samples_leaf because n=300 is small
param_dist = {
    'n_estimators': [500, 1000],
    'max_features': ['sqrt', 'log2', 0.5],
    'min_samples_leaf': [2, 5, 10, 15], 
}

tscv = TimeSeriesSplit(n_splits=5)
qrf = RandomForestQuantileRegressor(random_state=42)

# --- 3. Run Tuning for CRPS ---
tuned_search = RandomizedSearchCV(
    estimator=qrf,
    param_distributions=param_dist,
    scoring=crps_scorer,
    cv=tscv,
    n_iter=15,
    n_jobs=-1
)

tuned_search.fit(X_train, y_train)

# --- 4. Comparison Logic ---
# Default Model (Meinshausen 2006)
default_qrf = RandomForestQuantileRegressor(
    n_estimators=1000, 
    max_features='sqrt', 
    min_samples_leaf=5, 
    random_state=42
).fit(X_train, y_train)

# Best Tuned Model
best_qrf = tuned_search.best_estimator_

print(f"Best Params for CRPS: {tuned_search.best_params_}")