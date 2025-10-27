from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#do 3,6,9 and 12 month forecasts-> long and short term
#compare with out of sample method-> data got up to then model's recursive forecasting vintages, find the corresponding benchmark vintage

path = 'Code/Data/Cleaned_Data/QRF_data.csv'
df = pd.read_csv(path, index_col='Date', parse_dates=True)

#look at part where all values exist:
start_date = '2001-04-01'
end_date = '2025-04-01'
df = df.loc[start_date:end_date]
df.info()

#Core_CPI is an object-> coerce to float
df['Core_CPI'] = pd.to_numeric(df['Core_CPI'], errors='coerce') 


#split into train and test set to avoid data leakage

#not adding covid dummies because would be cheating as motivation is to find a model that better copes with the post 2020 period

















#check for NaNs
nans_per_column = df.isna().sum()
print(nans_per_column)

#check when NaNs occured:
print(df[df['Core_CPI'].isna()])
print(df[df['unemployment_rate'].isna()])
print(df[df['Wage_change'].isna()])


#Handling NaNs Timeseries appropriately wit linear interpolation 
#df['unemployment_rate'] = df['unemployment_rate'].interpolate(method='linear', limit_direction='both')
#df['Wage_change'] = df['Wage_change'].interpolate(method='linear', limit_direction='both')

#nans_per_column = df.isna().sum()
#print(nans_per_column)