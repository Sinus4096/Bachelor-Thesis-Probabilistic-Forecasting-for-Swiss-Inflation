from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima_process import ArmaProcess 
from statsmodels.tsa.stattools import adfuller


#do 3,6,9 and 12 month forecasts-> long and short term
#compare with out of sample method-> data got up to then model's recursive forecasting vintages, find the corresponding benchmark vintage
#need to extract point forecast if want to compare with forecast of snb???
#add dummy for covid? some motnhs before will know when have things closed
#ai studios says train data til 2006 und gemini til 2009-> between
#default facories fürs modeln
#hyperparameter with optuna

'''Recursive Updating Scheme:
Frequency: The paper updates quarterly (every three months). You should follow this, as inflation dynamics often have a quarterly rhythm, and it makes computational sense.
Mechanism:
Vintage 1: Train your model using data from January 2001 up to your chosen initial cutoff (e.g., December 2005). Generate forecasts for horizons t+3, t+6, t+9, t+12 months (i.e., for Q1, Q2, Q3, Q4 2006).
Vintage 2: Add one quarter of data (January-March 2006). Retrain the model using data from January 2001 up to March 2006. Generate forecasts for Q2, Q3, Q4 2006 and Q1 2007.
Continue: Repeat this process, adding three months of data to your training set and re-estimating the model, until your evaluation sample (up to December 2025) is exhausted.
Out-of-Sample Evaluation Period:
This will start after your initial training period. If you train until December 2005, your first out-of-sample forecasts will be for early 2006.
Your evaluation sample will range from January 2006 (or wherever your first forecast begins) until December 2025.'''



path = 'Code/Data/Cleaned_Data/QRF_data.csv'
df = pd.read_csv(path, index_col='Date', parse_dates=True)

#look at part where all values exist:
start_date = '2000-05-01'
end_date = '2025-04-01'
df = df.loc[start_date:end_date]
df.info()

#Core_CPI is an object-> coerce to float
df['Core_CPI'] = pd.to_numeric(df['Core_CPI'], errors='coerce') 

#check for NaNs
nans_per_column = df.isna().sum()
print(nans_per_column)
#check which years are missing
print(df[df['infl_e_current_year'].isna()])
#only in year 2000-> can ignore as taking yearly change so will use the observation in year 2000 eitherway


#analyze the two y variables Core and Headline CPI:
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10), sharex=True)

axes[0].plot(df.index, df['Core_CPI'], label='Core CPI', linewidth=2, color='C0')
axes[0].set_ylabel('CPI Index')
axes[0].legend()
axes[0].grid(True, linestyle=':', alpha=0.7)

axes[1].plot(df.index, df['Headline_CPI'], label='Headline CPI', linewidth=2, color='C1')
axes[1].set_ylabel('CPI Index')
axes[1].legend()
axes[1].grid(True, linestyle=':', alpha=0.7)

fig.suptitle('Time Series of Core and Headline CPI', fontsize=16)
axes[1].set_xlabel('Date') 
plt.tight_layout()
plt.subplots_adjust(top=0.95) 
plt.show()
#see clear trend 
#variance of the series increases with level-> heteroscedasticity

#plot ACF and PACF
cols_to_analyse = ['Core_CPI', 'Headline_CPI']
titles = ['Core CPI', 'Headline CPI']
lags_to_show = 48
for i, col in enumerate(cols_to_analyse):
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))
    plot_acf(df[col], lags=lags_to_show, ax=axes[0], title='Autocorrelation (ACF)')
    axes[0].grid(True, linestyle=':', alpha=0.7)
    plot_pacf(df[col], lags=lags_to_show, ax=axes[1], title='Partial Autocorrelation (PACF)')
    axes[1].grid(True, linestyle=':', alpha=0.7)
    fig.suptitle(f'ACF & PACF for {titles[i]}', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
plt.show()

#both ACF's show slow linear decay with all lags being highly significant; is clear indice for strong non-stationary trend




#generally why use log: 1. right skewness, 2. stabilize variance, 3. linearize relationship, here we want to stabilizy variance
#to detrend: seasonal differencing-> get year-over-year inflation rates
#shift is fine as we wouldnt have data for the inflation rates outside eitherway
for col in cols_to_analyse:
    df[col] = np.log(df[col]).diff(12)
df.head()

start_date = '2001-05-01'
end_date = '2025-04-01'
df = df.loc[start_date:end_date]
#check for NAN again -> all fine
nans_per_column = df.isna().sum()
print(nans_per_column)

#if look at plots seem to oscillate around a roughly 0 mean condition-> trend should be gone
#ACF: now a faster, rather geometric decay 
#PACF: sharp cutoff at lag 1-> probably a non-stationary RW but already try ADF test (AR(1) process)
for col in cols_to_analyse:
    adf_result = adfuller(df[col], autolag='AIC')
    print(f"Results for: {col}")
    print(f'p-value: {adf_result[1]:.4f}')    
    if adf_result[1] <= 0.05:
        print("Result: STATIONARY")
    else:
        print("Result: NON-STATIONARY")
#try first difference to get ridd off the RW 




#split into train and test set to avoid data leakage

#not adding covid dummies because would be cheating as motivation is to find a model that better copes with the post 2020 period


#split into one df for core inflation and one for headline inflation

#take log then yearly changes depending on variables maybe also from them






#stabilize the variance





#Handling NaNs Timeseries appropriately wit linear interpolation 
#df['unemployment_rate'] = df['unemployment_rate'].interpolate(method='linear', limit_direction='both')
#df['Wage_change'] = df['Wage_change'].interpolate(method='linear', limit_direction='both')

#nans_per_column = df.isna().sum()
#print(nans_per_column)