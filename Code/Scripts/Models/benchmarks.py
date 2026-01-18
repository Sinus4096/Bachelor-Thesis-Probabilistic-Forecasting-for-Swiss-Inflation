import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from scipy import stats

#load data
path='Code/Data/Cleaned_Data/data_stationary.csv' 
df=pd.read_csv(path, parse_dates=['Date'])

#get target cols
target_cols =[col for col in df.columns if col.startswith('target')]

# Define the quantile levels
key_quantiles=[0.05, 0.16, 0.50, 0.84, 0.95]  #for evaluation and plotting
dense_quantiles =np.linspace(0.01, 0.99, 99)   #for fan charts and CRPS

#initialize results storage
results_storage=[]
dense_data={}

#main loop over target cols
for target_col in target_cols:

