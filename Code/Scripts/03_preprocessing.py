from pathlib import Path
import pandas as pd
import numpy as np


#import data
path ='Code/Data/Cleaned_Data/data_merged.csv'
df=pd.read_csv(path, index_col='Date', parse_dates=True)

#define forecast horizons
horizons=[3, 6, 9, 12]
#----------------------------
#make stationary according to 02_eda_raw.py
#-------------------------------------------
#initialize a DF
df_stationary=pd.DataFrame(index=df.index)
#1. use formulat 12/h) * [ln(P_t) - ln(P_{t-h})] * 100 on CPI-variables e.g. for h=12: 12-month YoY growth rate
for h in horizons:
    df_stationary[f'target_headline_{h}m'] =(12/h)*(np.log(df['Headline_CPI']).diff(h))* 100
    df_stationary[f'target_core_{h}m']=(12/h) *(np.log(df['Core_CPI']).diff(h))*100
#2. take log % growth 
vars_to_log_diff = ['gdp_index_ch', 'gdp_index_eu', 'PPI', 'real_turnover', 'retail_turnover', 'Manufacturing_EU', 'Vol_loans', 
                    'Exchange_Rate_CHF'] 
df_stationary[vars_to_log_diff] = np.log(df[vars_to_log_diff]).diff(1)*100
#3. add all other variables (which were already stationary or in a rate (wage change))
#subtract the current columns from the original dataframe's columns
already_processed=set(df_stationary.columns)  #all cols already in df_stationary
all_columns= set(df.columns)       #all cols to process (are in the original df)
remaining_vars= list(all_columns-already_processed)    #difference: which not already in df_stationary
#loop through remaining vars and add them 
for var in remaining_vars:
    df_stationary[var]= df[var]
#check
print(df_stationary.head())

#through differencing first year have no data for core and headline cpi anymore (-> which is why we didnt drop the inflation expectations
#during data ingestion)-> set new starting date at one year later:
start_date ='2001-05-01'
#end so have the latest months
df_stationary=df_stationary.loc[start_date:] 

#check again
df_stationary.head()
#look if no NA's anymore:
nans_per_column =df_stationary.isna().sum()
print(nans_per_column)

#-> yes no NA's

#---------------------------
#add lags
#--------------------------
#not split into 2 df (1 for each y-variable) yet as Headline CPI includes food and energy, which are volatile. However, spikes in energy prices
#(Headline) often "leak" into Core CPI a few months later (e.g., higher transport costs eventually make clothes and services more expensive). -> 
#might be better predictors than core cpi lags, but the lags will probably be correlated-> need to be cautious in the feature selection part

#use 1-month growth rates (pi_t^1) as autoregressive lags
df_stationary['headline_1m']=np.log(df['Headline_CPI']).diff(1)*100
df_stationary['core_1m']=np.log(df['Core_CPI']).diff(1)*100
#add 1- and 2-month lags
for i in [1, 2]:
    df_stationary[f'headline_lag_{i}'] = df_stationary['headline_1m'].shift(i)
    df_stationary[f'core_lag_{i}'] = df_stationary['core_1m'].shift(i)
#keep NA's for the lagged variables as they are needed for prediction later



#--------------------------
#add Monthly Cycle features
#--------------------------------
#why dummies and not sine/cosine?: Swiss data, Monthly Dummies are preferred over smooth Sine/Cosine waves because 
#they better capture sharp seasonal spikes.
#convert month to dummies
dummies=pd.get_dummies(df_stationary.index.month, prefix='Month').astype(int)
dummies.index=df_stationary.index   #ensure same index
df_stationary=pd.concat([df_stationary, dummies], axis=1)   #add to df_stationary
#check
print(df_stationary.head())
#---------------------------
#create forecast horizons / shift targets
#----------------------------------------
#reason: need data of today to predict three months ahead-> features at t align with the target realized at t+h
for h in horizons:
    df_stationary[f'target_headline_{h}m'] = df_stationary[f'target_headline_{h}m'].shift(-h)
    df_stationary[f'target_core_{h}m'] = df_stationary[f'target_core_{h}m'].shift(-h)
#see if worked:
print(df_stationary.tail())
#do not drop NA's here: need the bottom rows later to predict the "future"



#--------------------------------------
#save the DF
#----------------------------------------
#define path to csv directory
CODE_DIR=Path(__file__).parent.parent
output_path=CODE_DIR /"Data"/"Cleaned_Data"
#print the processed df to outputpath
output_file=output_path/'data_stationary.csv'
df_stationary.to_csv(output_file, index=True)


