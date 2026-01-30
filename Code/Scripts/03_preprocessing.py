from pathlib import Path
import pandas as pd
import numpy as np
#preprocessing done according to 02_eda_raw.py

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
#1. use formulat 12/h)*[ln(P_t)-ln(P_{t-h})]*100 on CPI-variables e.g. for h=12: 12-month YoY growth rate
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

#look if no NA's:
nans_per_column =df_stationary.isna().sum()
print(nans_per_column)

#through differencing for h=12, will have NA till may 2001 in beginning but will shift-> will ignore 
#NA of inflation expectations will be dropped later as we'll have other NA's when creating lags
#->looks good for now

#---------------------------
#add lags
#--------------------------
#not split into 2 df (1 for each y-variable) yet as Headline CPI includes food and energy, which are volatile. However, spikes in energy prices
#(Headline) often "leak" into Core CPI a few months later (e.g., higher transport costs eventually make clothes and services more expensive). -> 
#might be better predictors than core cpi lags, but the lags will probably be correlated-> need to be cautious in the feature selection part

#use 1-month growth rates (pi_t^1) as autoregressive lags
df_stationary['headline_1m']=np.log(df['Headline_CPI']).diff(1)*100
df_stationary['core_1m']=np.log(df['Core_CPI']).diff(1)*100
#add 1- and 2-month lags, two lags only because of PACF plots in 02_eda_raw.py: go down rapidly after lag 2
for i in [1, 2]:
    df_stationary[f'headline_lag_{i}']=df_stationary['headline_1m'].shift(i)
    df_stationary[f'core_lag_{i}']= df_stationary['core_1m'].shift(i)
#keep NA's for the lagged variables as they are needed for prediction later



#--------------------------
#add Cycle features: sine/consine transformations
#--------------------------------
#are better than adding monthly dummies for bvar model as they do not eat up so many degrees of freedom
period = 12
#get the month index (1-12)
month_idx=df_stationary.index.month

#generate 2 Fourier pairs->capture the main annual and semi-annual cycles
for i in range(1, 3):  
    df_stationary[f'sin_cycle_{i}']= np.sin(2* np.pi*i*month_idx/period)
    df_stationary[f'cos_cycle_{i}'] = np.cos(2* np.pi*i*month_idx /period)
#check
print(df_stationary.head())
#---------------------------
#create forecast horizons / shift targets
#----------------------------------------
#reason: need data of today to predict three months ahead-> features at t align with the target realized at t+h
for h in horizons:
    df_stationary[f'target_headline_{h}m']= df_stationary[f'target_headline_{h}m'].shift(-h)
    df_stationary[f'target_core_{h}m']= df_stationary[f'target_core_{h}m'].shift(-h)
#see if worked:
print(df_stationary.tail())


#-----------------------------
#cope with NA's
#-----------------------------
#chekc for NA's
nans_per_column_yoy =df_stationary.isna().sum()
print(nans_per_column_yoy)
#NA of lags because cannot make lags of data that not exist-> check if somewhere else:
rows_with_nan=df_stationary[df_stationary.isna().any(axis=1)]
print(rows_with_nan)
#->NA lags are as expected in the first two rows-> set start date later to avoid crash of qrf
#do not drop NA's of targets later here: their availability depends on the forecast horizon
#as will evaluate using yoy inflation dropping later won't make a change
start_date=df_stationary['infl_e_current_year'].first_valid_index()
df_stationary= df_stationary.loc[start_date:] 
#recheck
rows_with_nan=df_stationary[df_stationary.isna().any(axis=1)]
print(rows_with_nan)
#good: only ones left with missing targets





#----------------------------------
#YoY %-growth rates for comparison
#----------------------------------
#create reference for actual realized YoY inflation-> will compare predictions against this
df_yoy = pd.DataFrame(index=df.index)
df_yoy['Headline']=np.log(df['Headline_CPI']).diff(12)*100 
df_yoy['Core'] =np.log(df['Core_CPI']).diff(12)*100

#add levels for Conditional forecast later
df_yoy['Headline_level']= df['Headline_CPI']
df_yoy['Core_level']= df['Core_CPI']

#same starting date as df_stationary
start_date ='2001-05-01'
df_yoy=df_yoy.loc[start_date:] 

#check for NA's
nans_per_column_yoy =df_yoy.isna().sum()
print(nans_per_column_yoy)

#--------------------------------------
#save the DF's
#----------------------------------------
#define path to csv directory
CODE_DIR=Path(__file__).parent.parent
output_path=CODE_DIR /"Data"/"Cleaned_Data"
#print the processed df to outputpath
output_file1=output_path/'data_stationary.csv'
df_stationary.to_csv(output_file1, index=True)
output_file2=output_path/'data_yoy.csv'
df_yoy.to_csv(output_file2, index=True)