#for feature selection of QRF:

#2. A Warning: The "Lag Correlation" Trap
#In CPI data, headline_lag_1 and headline_lag_2 (and even core_lag_1) are often highly correlated with each other (multicollinearity).
#Standard RF feature importance (Gini/Impurity) tends to spread importance across correlated features, making them all look "okay" rather than one looking "great."
#Recommendation: Use Permutation Importance or SHAP values on your Random Forest. They are much more reliable for time-series features that move together.
#3. Will the importance be the same for all quantiles?
#This is the only nuance. Sometimes a feature might not be great at predicting the median (50th percentile) but is very good at predicting extreme tails (e.g., the 5th or 95th percentile).
#Example: An energy price spike might not change the most likely inflation outcome much, but it might significantly increase the "upside risk" (the 95th percentile).
#The fix: Use the RF to filter out the "garbage" features (the ones that clearly provide no value), but keep the top 10–15 features even if they aren't "stars," just in case they help define the tails of the distribution in the QRF.



#or should not do i t? because: While your intuition is correct that 25 features for ~250 observations is a lot, doing a separate feature 
# importance analysis before the modeling step introduces two major risks: Data Leakage and Model Mismatch.