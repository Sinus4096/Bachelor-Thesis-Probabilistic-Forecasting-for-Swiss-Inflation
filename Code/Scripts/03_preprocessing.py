from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
#preprocessing done according to 02_eda_raw.py (findings are based on training data but applied to whole data to prevent look-ahead bias)

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
vars_to_log_diff= ['PPI', 'real_turnover', 'retail_turnover', 'Manufacturing_EU', 'Vol_loans', 'Exchange_Rate_CHF'] 
df_stationary[vars_to_log_diff] = np.log(df[vars_to_log_diff]).diff(1)*100
#need to do quarterly for gdp indices as we forward filled them-> do 3-month diff
gdp_vars=['gdp_index_ch', 'gdp_index_eu']
df_stationary[gdp_vars]= np.log(df[gdp_vars]).diff(3)*100

#3. add all other variables (which were already stationary or in a rate (wage change))
#subtract the current columns from the original dataframe's columns
already_processed=set(df_stationary.columns)  #all cols already in df_stationary
all_columns= set(df.columns)       #all cols to process (are in the original df)
#make sure core and headline cpi are not in the remaining vars as they are already processed
exclude_from_features ={'Headline_CPI', 'Core_CPI'}
remaining_vars=list(all_columns -already_processed - exclude_from_features) #difference: which not already in df_stationary
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
#df_stationary.drop(columns=['headline_1m', 'core_1m'], inplace=True)  #drop the 1-month growth rates as we only want the lags as features

#--------------------------
#add Cycle features: sine/consine transformations
#--------------------------------
#are better than adding monthly dummies as do not eat up so many degrees of freedom
period= 12
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
#create bvar df without lags as bvar creates its own lags, otherwise bvar will use lags to predict lags
#-----------------------------
df_bvar=df_stationary.copy()
#drop lagged variables and cycle features
lagged_cols=[col for col in df_bvar.columns if 'lag_' in col]  #leave autoregressive features so bvar can make lags based on them
df_bvar.drop(columns=lagged_cols, inplace=True)

#for bvar will leave NA for now and start training later as 12. lag will be created by the model itself-> set start date to latest
#available date of predictors :
start_date=df_bvar['infl_e_current_year'].first_valid_index()
df_bvar= df_bvar.loc[start_date:] 



#cols_to_drop=['real_turnover', 'retail_turnover', 'kofbarometer', 'Manufacturing_EU', 'Business_Confidence_EU',  'headline_lag_1', 'core_lag_1']
#df_stationary.drop(columns=cols_to_drop, inplace=True)


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
#set start date as last available date of predictors +1 year as bvar will have to create a 12 month lag for it too
base_date=df_stationary['infl_e_current_year'].first_valid_index()
#add 12 months to that date 
start_date= base_date+pd.DateOffset(months=12)
df_stationary= df_stationary.loc[start_date:] 
#recheck
rows_with_nan=df_stationary[df_stationary.isna().any(axis=1)]
print(rows_with_nan)
#good: only ones left with missing targets


#------------------------------------
#look at stats-> need to standardize for qrf?
#-------------------------
print(df_stationary.describe().T)

#----------------------
#standardize features
# --------------------------------
#for feature importance of qrf and for bvar generallyneed to standardize the features but not the targets because want to keep them
#in original units for evaluation later on

#need to standardize based on data we have available: meaning based on training data to prevent look-ahead bias
#define split date
test_start_date= '2013-07-01'
#define exclusion criteria: dont want to standardize targets 
#vars_to_scale =[col for col in df_stationary.columns if 'target' not in col]
vars_to_scale_bvar=[col for col in df_bvar.columns if 'target' not in col]  

#def scaler
#scaler=StandardScaler()
#fit scale on training data only
#scaler.fit(df_stationary.loc[:test_start_date].iloc[:-1][vars_to_scale]) 
#transform the data
#df_stationary[vars_to_scale]= scaler.transform(df_stationary[vars_to_scale])
#new scaler for BVAR dataframe 
#scaler_bvar= StandardScaler()
#fit only on BVAR training data 
#scaler_bvar.fit(df_bvar.loc[:test_start_date].iloc[:-1][vars_to_scale_bvar])
#df_bvar[vars_to_scale_bvar]= scaler_bvar.transform(df_bvar[vars_to_scale_bvar])

#recheck descriptive stats
print(df_stationary.describe().T)
#decision: to have best fits: will do scaling recursively in the training loop of the qrf model





#----------------------------------------
#for 2. configuration of the stationary data: direct 12-month YoY growth rate prediction (no 12/h scaling)
#-----------------------------------------------------------
df_stationary2=df_stationary.copy()
#drop previous targets
target_cols_to_drop= [col for col in df_stationary2.columns if 'target_' in col]
df_stationary2.drop(columns=target_cols_to_drop, inplace=True)

#12 month yoy growth rates in %
yoy_headline_raw=(np.log(df['Headline_CPI']).diff(12))*100
yoy_core_raw= (np.log(df['Core_CPI']).diff(12))* 100

#shift according to horizons
for h in horizons:
    df_stationary2[f'target_headline_{h}m'] = yoy_headline_raw.shift(-h)
    df_stationary2[f'target_core_{h}m'] = yoy_core_raw.shift(-h)







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
output_file3= output_path/'data_stationary_direct.csv'
df_stationary2.to_csv(output_file3, index=True)
output_file4= output_path/'data_stationary_bvar.csv'
df_bvar.to_csv(output_file4, index=True)
