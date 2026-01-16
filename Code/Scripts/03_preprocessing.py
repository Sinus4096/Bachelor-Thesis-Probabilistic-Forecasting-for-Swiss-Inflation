from pathlib import Path
import pandas as pd
import numpy as np


#import data
path ='Code/Data/Cleaned_Data/data_merged.csv'
df=pd.read_csv(path, index_col='Date', parse_dates=True)

#----------------------------
#make stationary according to 02_eda_raw.py
#-------------------------------------------

#1.YoY %-changes for the two CPI-variables
df_stationary= np.log(df[['Core_CPI', 'Headline_CPI']]).diff(12)*100 

#2. take log % growth for the trending variables
trending_vars=['gdp_index_ch', 'gdp_index_eu', 'PPI', 'real_turnover', 'retail_turnover', 'Manufacturing_EU', 'Vol_loans']
df_stationary[trending_vars]=np.log(df[trending_vars]).diff().dropna()*100

#3. do simple differencing for other non stationary variables: 
non_s_vars= ['Wage_change','Exchange_Rate_CHF','M3_change']
#take differences of those 
df_stationary[non_s_vars] = df[non_s_vars].diff().dropna()

#4. add all other variables (which were already stationary)
#subtract the current columns from the original dataframe's columns
already_processed=set(df_stationary.columns)  #all cols already in df_stationary
all_columns= set(df.columns)       #all cols to process (are in the original df)
remaining_vars= list(all_columns-already_processed)    #difference: which not already in df_stationary
#loop through remaining vars and add them 
for var in remaining_vars:
    df_stationary[var]= df[var]
#check
df_stationary.head()

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

#create lag veriables for 1, 3 and 6 months: 1 month before probably most important, 3 for quartely dymamics and 6 for half year dynamics
lags_to_keep = [1, 3, 6] 

for i in lags_to_keep:
    df_stationary[f'headline_lag_{i}']=df_stationary['Headline_CPI'].shift(i)
    df_stationary[f'core_lag_{i}']=df_stationary['Core_CPI'].shift(i)

#drop the rows that now have NaNs 
df_stationary= df_stationary.dropna()



#--------------------------
#add sine and cosine transformations
#--------------------------------

#calculate them now, but config file will decide later whether to use them or drop them.

df_stationary['month'] =df_stationary.index.month #define month column from date index
#add sine and cosine transformations of the Annual Cycle (->/12)
df_stationary['month_sin']= np.sin(2*np.pi*df_stationary['month'] /12)
df_stationary['month_cos']= np.cos(2*np.pi*df_stationary['month']/12)
# Drop the raw 'month' column if you only want the cyclic encoding
df_stationary = df_stationary.drop(columns=['month'])


#---------------------------
#create forecast horizons / shift targets
#----------------------------------------
#reason: need data of today to predict three months ahead

#create targets for 3, 6, 9, and 12 months ahead
horizons= [3, 6, 9, 12]

for h in horizons:
    df_stationary[f'target_headline_{h}m']= df_stationary['Headline_CPI'].shift(-h)
    df_stationary[f'target_core_{h}m']= df_stationary['Core_CPI'].shift(-h)

#see if worked:
df_stationary.tail()
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


