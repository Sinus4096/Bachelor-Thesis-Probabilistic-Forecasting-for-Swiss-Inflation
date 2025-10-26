from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

path = 'Code/Data/Cleaned_Data/QRF_data.csv'
df = pd.read_csv(path, index_col='Date', parse_dates=True)

#look at part where all values exist:
start_date = '2001-02-01'
end_date = '2025-04-01'
df = df.loc[start_date:end_date]
df.info()

#Core_CPI is an object-> coerce to float
df['Core_CPI'] = pd.to_numeric(df['Core_CPI'], errors='coerce') 


#split into train and test set to avoid data leakage



















#check for NaNs
nans_per_column = df.isna().sum()
print(nans_per_column)

#Handling NaNs Timeseries appropriately wit linear interpolation 
df['unemployment_rate'] = df['unemployment_rate'].interpolate(method='linear', limit_direction='both')
df['Wage_change'] = df['Wage_change'].interpolate(method='linear', limit_direction='both')

nans_per_column = df.isna().sum()
print(nans_per_column)