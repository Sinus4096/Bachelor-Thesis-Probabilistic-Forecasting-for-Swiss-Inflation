import numpy as np
import pandas as pd
from quantile_forest import RandomForestQuantileRegressor
from sklearn.model_selection import TimeSeriesSplit


#------------------------------------------------
#Meinshausen Default Model
#------------------------------------------------


#prepare data: time series split to avoid shuffling 
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# 2. Initialize the QRF model
# Meinshausen (2006) recommends min_samples_leaf=5 for QRF
qrf = RandomForestQuantileRegressor(
    n_estimators=1000,       # Meinshausen used 1000 for stability
    max_features='sqrt',     # Standard for high-dimensional inflation datasets
    min_samples_leaf=5,      # Meinshausen's suggested default
    random_state=42,
    n_jobs=-1                # Use all CPU cores
)

# 3. Fit the model
qrf.fit(X_train, y_train)

# 4. Expanded Quantiles for Density/Risk Assessment
# We use a wider range to capture the "tails" of inflation
quantiles = [0.05, 0.16, 0.50, 0.84, 0.95]
predictions = qrf.predict(X_test, quantile=quantiles)

# 5. Extracting Insights
df_preds = pd.DataFrame(predictions, columns=[f'q{int(q*100)}' for q in quantiles])
median_forecast = df_preds['q50']
inflation_uncertainty = df_preds['q84'] - df_preds['q16'] # Inter-quantile range
upside_risk = df_preds['q95'] - df_preds['q50']         # Distance to upper tail




#-----------------------------------------------------
#Optimized Model with Hyperparametertuning
#-----------------------------------------------------

