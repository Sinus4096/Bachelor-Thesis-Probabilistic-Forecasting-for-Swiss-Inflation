from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima_process import ArmaProcess 
from statsmodels.tsa.stattools import adfuller
import math
from tabulate import tabulate


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

'''Since QRF doesn't inherently "know" the order of time, you must explicitly create lags of your target variable ($\pi_{t-1}, \pi_{t-2}$) and other predictors to capture autoregressive dynamics.
'''
'''core Inflation (HICPex): Often exhibits stronger non-linear relationships, particularly with inflation expectations. Studies show QRF is especially competitive and often more accurate for Core than for Headline inflation.

Headline Inflation (HICP): Driven heavily by global commodity prices (energy/food), which typically follow more linear dynamics. While QRF is still a valid tool here, linear models are often very competitive for Headline because these volatile external shocks can "overshadow" the mild non-linearities found in core components.'''

#this script serves as justification for further transformations

path ='Code/Data/Cleaned_Data/data_merged.csv'
df=pd.read_csv(path, index_col='Date', parse_dates=True)


#----------------------------------------
#Time Series Plots for Visual Analysis
#-------------------------------------------

#analyze the two y variables Core and Headline CPI:

#set style
plt.style.use('seaborn-v0_8-whitegrid')
#create a grid
fig, axes=plt.subplots(nrows=2, ncols=1, figsize=(12, 10), sharex=True, gridspec_kw={'hspace': 0.1})

#define color palette
core_color='#2E5A88'      
headline_color='#D97B42'  
#core cpi plot
axes[0].plot(df.index, df['Core_CPI'], label='Core CPI', linewidth=2.5, color=core_color)
#fill subtle
axes[0].fill_between(df.index, df['Core_CPI'], color=core_color, alpha=0.1) 
axes[0].set_ylabel('Index Value', fontweight='bold', labelpad=10)
axes[0].legend(frameon=True, loc='upper left')

#headline cpi plot
axes[1].plot(df.index, df['Headline_CPI'], label='Headline CPI', linewidth=2.5, color=headline_color)
#fill subtle
axes[1].fill_between(df.index, df['Headline_CPI'], color=headline_color, alpha=0.1) 
axes[1].set_ylabel('Index Value', fontweight='bold', labelpad=10)
axes[1].legend(frameon=True, loc='upper left')

#nicer styling
for ax in axes:
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax.grid(False, axis='x')

#set titles and labels
fig.suptitle('Consumer Price Index (CPI) Trends', fontsize=20, fontweight='bold', y=0.98)
axes[1].set_xlabel('Date', fontweight='bold', labelpad=10)

#plot
plt.tight_layout(rect=[0, 0, 1, 0.96]) 
plt.show()
#observations:
#there is a clear long-term upward trend in both Core and Headline CPI, showing a steady increase in price levels from 2000 through 2024.
#the Headline CPI exhibits more frequent, jagged fluctuations compared to the Core CPI, likely reflecting the volatile nature of food and energy
#prices which are excluded from the core metric. Additionally, both charts show a noticeable acceleration in the growth rate starting around 2021,
#indicating a period of sharper inflation in recent years.




#time series grid for predictors:

#define X variables
targets =['Core_CPI', 'Headline_CPI']
x_variables =[col for col in df.columns if col not in targets]

#split into 2 chunks
halfway =math.ceil(len(x_variables)/2)
chunks=[x_variables[:halfway], x_variables[halfway:]]

#loop through chunks
for fig_idx, chunk in enumerate(chunks):
    #def grid of plots
    n_cols=3
    n_rows =math.ceil(len(chunk) /n_cols)
    
    #create figure
    fig, axes =plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(18, n_rows*3.5), constrained_layout=True)
    #flatten the axis and define color
    axes_flat =axes.flatten()
    x_color='#4C72B0'

    for i, var_name in enumerate(chunk):
        ax=axes_flat[i]
        
        #drop the na's (before 2001) and plot
        series =df[var_name].dropna()
        ax.plot(series.index, series.values, label=var_name, linewidth=2, color=x_color)
        ax.fill_between(series.index, series.values, color=x_color, alpha=0.1)      #fill underneath the line
        
        #style
        ax.set_title(var_name, fontweight='bold', fontsize=13, loc='left', pad=10)
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        #clean
        for spine in ['top', 'right', 'left']:
            ax.spines[spine].set_visible(False)
        #make nicer: grid and legend
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
        ax.legend(frameon=True, loc='upper left', fontsize='small')

    #delete the created subplots we dont use
    for j in range(len(chunk), len(axes_flat)):
        fig.delaxes(axes_flat[j])

    #figure title for both chunks
    fig.suptitle(f'Economic Indicators Trends (Part {fig_idx +1})', fontsize=22, fontweight='bold')

    plt.show()



#observations:

#1. Variables like gdp_index, PPI, and Vol_loans exhibit clear deterministic or stochastic trends. I will apply log-differences 
# ($\Delta \ln(x)$) to convert these into growth rates to prevent spurious regression in the BVAR and ensure the QRF can handle 
# future values.

#2. Survey indicators (kofbarometer) and variables already expressed as changes (M1_change) appear mean-reverting. I will verify 
# stationarity with an ADF test; if $p < 0.05$, I will use them in their current form to preserve the signals.

#3. Interest rate variables (Saron_Rate) show structural breaks and long periods of zero-volatility. I will check for 'Unit Roots' 
# carefully here, as the standard ADF test can sometimes be misled by structural shifts.







#---------------------------------------
#Check For Correlation
#-----------------------------------------

#check which X variables are linearly related to the Inflation measures

#calc correlations of all X vars against the Y vars
corrs_with_core =df[x_variables].corrwith(df['Core_CPI']).sort_values(ascending=False)
corrs_with_headline =df[x_variables].corrwith(df['Headline_CPI']).sort_values(ascending=False)

#plot side by side bars
fig, ax= plt.subplots(figsize=(14, 8))
indices =np.arange(len(x_variables))
width=0.35

#plot bars
rects1=ax.bar(indices -width/2, corrs_with_core[x_variables], width, label='vs Core CPI', color='#2E5A88')
rects2 = ax.bar(indices+width/2, corrs_with_headline[x_variables], width, label='vs Headline CPI', color='#D97B42')

#make nicer
ax.set_title('Correlation of X Variables with Inflation Targets', fontsize=16, fontweight='bold')
ax.set_ylabel('Correlation Coefficient (Pearson)', fontweight='bold')
ax.set_xticks(indices)
ax.set_xticklabels(x_variables, rotation=90)
ax.legend()
ax.grid(True, axis='y', linestyle='--', alpha=0.5)
ax.axhline(0, color='black', linewidth=0.8)

plt.tight_layout()
plt.show()


#observations:

#1. "High-Signal" Predictors
#pos: gdp_index, real_turnover, retail_turnover, Vol_loans, and Manufacturing_EU show extremely high positive correlations (above 0.8)
#inverse: variable_mortgages, fin_spread, and CH_2int show strong negative correlations. These might be critical for your model to understand 
# downward pressure on inflation or structural economic shifts.

#2.spurious vs. real Relationships-> connection to time series plots
# gdp_index and Vol_loans have very high correlations, but the Trends Plot shows they are both strictly upward-trending.
#high correlation in non-stationary data is often just "two lines going up." -> if the correlation drops significantly after calculating
#growth rates, the initial relationship was largely driven by the time trend rather than a direct economic link.







#-------------------------------------------
#ACF and PACF plots
#-------------------------------------------

#want to check for all variables at once
all_vars=['Core_CPI', 'Headline_CPI'] +x_variables
#set colors
color_acf ='#2E5A88' 

#restrict number of figures to 5
vars_per_fig=4
#def chunknr for title
chunk_nr=1

#loop through variables in chunks
for start_idx in range(0, len(all_vars), vars_per_fig):
    
    #def current chunk
    chunk =all_vars[start_idx: start_idx+vars_per_fig]
    n_vars =len(chunk)    #nr of vars in this chunk (is not 4 if last chunk)
    
    # Create fig. squeeze=False ensures 'axes' is ALWAYS a 2D array [row, col]
    fig, axes=plt.subplots(nrows=n_vars, ncols=2, figsize=(10, n_vars*2.2), squeeze=False)
    
    #loop through vars in chunk
    for i, var_name in enumerate(chunk):
        #get data and drop NA's (before 2001)
        series=df[var_name].dropna()
        
        #acf plot
        plot_acf(series, ax=axes[i, 0], lags=40, color=color_acf, title=f'ACF: {var_name}', vlines_kwargs={"colors": color_acf})
        
        #pacf plot
        plot_pacf(series, ax=axes[i, 1], lags=40, color=color_acf, title=f'PACF: {var_name}', vlines_kwargs={"colors": color_acf})
        
        #styling
        for ax in axes[i]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, axis='y', linestyle='--', alpha=0.5)
            ax.set_ylim(-1.1, 1.1)
            ax.tick_params(axis='both', which='major', labelsize=10)
    #figure title for all chunks
    fig.suptitle(f'ACF & PACF Plots of all Variables (Part {chunk_nr})', fontsize=22, fontweight='bold')
    #plot
    plt.tight_layout()
    plt.show()
    #add 1 to chunknr for next chunk plot
    chunk_nr+=1



#observations:
#Core_CPI /Headline_CPI: The ACF shows a very slow linear decay, while the PACF has a sharp spike at lag 1; this is a sign of 
# non-stationarity. 
#kofbarometer: The ACF drops toward zero relatively quickly and even turns negative, suggesting this survey-based indicator is already 
# stationary.-> ADF test to verify but can likely use this in its raw level format.
# unemployment_rate: displays a very persistent ACF, indicating it is non-stationary and likely follows a random walk. -> probably use 
# the first difference (change in percentage points) for the models.
#oilprices: The ACF remains significantly above the confidence interval for over 30 lags, confirming non-stationarity.-> will likely have to
# transform into log-returns to capture price shocks rather than price levels.
#gdp_index_x /gdp_index_y: These show the strongest trend of all variables with almost no decay in the ACF.->must calculate quarterly 
# or yearly growth rates for these to be usable in the models.
#infl_e_current_year /infl_e_next_year: These expectations are quite persistent but show a faster ACF decay than the CPI itself. 
#However, to stay consistent with the inflation targets, using the change in expectations is probably best.
#Saron_Rate / CH_2int: Both show very high autocorrelation at nearly all lags, typical of interest rate levels in trending environments.
#->will likely use the first difference to model the change in monetary policy.
#fin_spread: The ACF decays more quickly than the interest rates themselves but remains above the threshold for many lags. -> wait for ADF
#test to make decision
#retail_turnover: The ACF remains significantly high for all 40 lags, indicating a strong trend and non-stationarity.-> will usse
# log-growth rates to model the percentage change in retail activity.
#real_turnover: Similar to retail, the ACF exhibits almost no decay, confirming it is non-stationary. -> also log-growth transformation
# Manufacturing_EU: Shows a persistent, slow-decaying ACF typical of a non-stationary structural trend. The PACF spike at lag 1 
# suggests an AR(1) process in the trend.
#PPI: The ACF stays above the confidence interval for the entire window, signifying non-stationarity. This reflects long-term price 
# level drift that needs to be converted into inflation rates.
#Vol_loans: This displays the highest persistence of all variables, with a nearly flat ACF at 1.0, indicating it is strongly non-stationary. 
# -> will use log-differences to capture the growth in credit volume.
#M1_change, M2_change, & M3_change: These show a much faster ACF decay compared to other variables, reaching zero around lag 15. While 
# they are "changes," they still exhibit some persistence;-> need to verify with an ADF test, as they may be near-stationary.
#Exchange_Rate_CHF: The ACF decays very slowly, a sign that exchange rate levels are non-stationary (random walk). -> probably use log returns
#variable_mortgages: Displays extreme persistence in the ACF, typical of "sticky" interest rate levels, making it non-stationary. 
# -> will likely use first differences to model mortgage rate adjustments.
#EU_fin_spread: The ACF decays notably faster than the base interest rates but remains significant for many lags, suggesting it is non-stationary
# or highly persistent. The PACF shows a significant spike at lag 1.
#Business_Confidence_EU: The ACF crosses into negative territory and oscillates, which is a common pattern for stationary or cyclical 
# indicators. -> make ADF test but might be able to use var in levels.
#Wage_change: Shows moderate persistence with the ACF dropping to near-zero after lag 15-20. Like the money supply changes, it is closer 
# to stationarity than the index variables.








#-----------------------------
#ADF tests
#-----------------------------


#fct for adf test
def run_adf_test(series, var_name):
    """adf test of all vars and return dic of results"""
    #drop NA's (-> still have some before 2001 but wont be there anymore after differencing)
    series =series.dropna()    
    #run test, AIC to find best lag lenth
    result =adfuller(series, autolag='AIC')    
    #create dic of results and return it
    return {'Variable': var_name, 'ADF Statistic': round(result[0], 4), 'p-value': round(result[1], 4), 'Lags Used (AIC for decision)': result[2],
        'Observations': result[3], 'Stationary? (at 5%)': 'Yes' if result[1] < 0.05 else 'No', '1% Critical': round(result[4]['1%'], 4),
        '5% Critical': round(result[4]['5%'], 4)}

#initialize list for resutls
adf_results=[]
#loop through all variables of df
for col in df.columns:
    #all cols shoul dbe float (from preprocessing) but still check to avoid errors
    if pd.api.types.is_numeric_dtype(df[col]):
        res=run_adf_test(df[col], col)
        adf_results.append(res)


#to df for nice tabular display
adf_table=pd.DataFrame(adf_results)
#display the table
print("\n Augmented Dickey-Fuller Test Results: ")
print(adf_table) 


#identified problem with AIC:
#AIC is great to pick "best" lag, but for M1/M2/M3 it's picking 15+ lags-> it means that the model is struggling to clear out the noise.
#Furthermore for monthly data with seasonality, the test can miss the 12th month if we don't allow it to look far enough.
# what to do? for specific vars difference and then redo acf, pacf and adf
# which vars? the ones with high ADF p-values and high lag counts, like retail_turnover (11 lags) and the M-changes (15-16 lags)
# (as these often hide seasonal patterns)









#---------------------------------
#Analysis for seasonality
#---------------------------------
#1. vars with strong trend:
#trend is so dominant that it hides seasonal signa, once calculate the log-growth ratem, trend is removed-> hidden seasonal spikes will become visible.
trending_vars=['Core_CPI', 'Headline_CPI', 'gdp_index_ch', 'gdp_index_eu', 'PPI', 'real_turnover', 'retail_turnover', 'Manufacturing_EU', 'Vol_loans']

#take log growth rates (*100 to have % changes)
df_growth=np.log(df[trending_vars]).diff().dropna()*100

#start the plotting to see if there is seasonal pattern now:
#set colors
color_acf ='#2E5A88' 
#restrict number of figures to 5
vars_per_fig=4
#def chunknr for title
chunk_nr=1

#loop through variables in chunks
for start_idx in range(0, len(trending_vars), vars_per_fig):
    
    #def current chunk
    chunk=trending_vars[start_idx: start_idx+vars_per_fig]
    n_vars =len(chunk)    #nr of vars in this chunk (is not 4 if last chunk)
    
    #create figure 
    fig, axes=plt.subplots(nrows=n_vars, ncols=2, figsize=(10, n_vars*2.2), squeeze=False)
    
    #loop through vars in chunk
    for i, var_name in enumerate(chunk):
        #get data and drop NA's (before 2001)
        series=df_growth[var_name].dropna()
        
        #acf plot
        plot_acf(series, ax=axes[i, 0], lags=40, color=color_acf, title=f'ACF: {var_name}', vlines_kwargs={"colors": color_acf})
        
        #pacf plot
        plot_pacf(series, ax=axes[i, 1], lags=40, color=color_acf, title=f'PACF: {var_name}', vlines_kwargs={"colors": color_acf})
        
        #styling
        for ax in axes[i]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, axis='y', linestyle='--', alpha=0.5)
            ax.set_ylim(-1.1, 1.1)
            ax.tick_params(axis='both', which='major', labelsize=10)
    #figure title for all chunks
    fig.suptitle(f'ACF & PACF Plots of Detrended Variables (Part {chunk_nr})', fontsize=22, fontweight='bold')
    #plot
    plt.tight_layout()
    plt.show()
    #add 1 to chunknr for next chunk plot
    chunk_nr+=1
    
#all look good except for core and headline cpi: have clear trend appearing in the acf plot-> will take YoY changes for the two:  
#first drop the two cpi cols bc want to replace them with their yoy changes 
df_growth=df_growth.drop(['Core_CPI', 'Headline_CPI'], axis=1)
#calculate yoy changes
horizons=[3, 6, 9, 12]
for h in horizons:
    df_growth[f'target_headline_{h}m'] =(12/h)*(np.log(df['Headline_CPI']).diff(h))* 100
    df_growth[f'target_core_{h}m']=(12/h) *(np.log(df['Core_CPI']).diff(h))*100

#check whether cols exist now:
df_growth.columns
#redo acf check for the two cols to check whether looks stationary now:
#define fig
fig, axes=plt.subplots(nrows=len(df_growth.columns), ncols=2, figsize=(16, 10))
#loop through the target variables
for i, var_name in enumerate(df_growth.columns):
        #get data and drop NA's
        series=df_growth[var_name].dropna()
        #acf plot
        plot_acf(series, ax=axes[i, 0], lags=40, color=color_acf, title=f'ACF: {var_name}', vlines_kwargs={"colors": color_acf})        
        #pacf plot
        plot_pacf(series, ax=axes[i, 1], lags=40, color=color_acf, title=f'PACF: {var_name}', vlines_kwargs={"colors": color_acf})
        
        #styling
        for ax in axes[i]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, axis='y', linestyle='--', alpha=0.5)
            ax.set_ylim(-1.1, 1.1)
            ax.tick_params(axis='both', which='major', labelsize=10)
#figure title for all chunks
fig.suptitle(f'ACF & PACF Plots of YoY %-Changes of CPI Variables', fontsize=22, fontweight='bold')
#plot
plt.tight_layout()
plt.show()

#do adf test
adf_results=[]
#loop through all variables of df
for col in df_growth.columns:
    #all cols shoul dbe float (from preprocessing) but still check to avoid errors
    if pd.api.types.is_numeric_dtype(df_growth[col]):
        res=run_adf_test(df_growth[col], col)
        adf_results.append(res)

#to df for nice tabular display
adf_table=pd.DataFrame(adf_results)
#display the table
print("\n Augmented Dickey-Fuller Test Results for Trending Variables: ")
print(adf_table) 
#remark: we saw that diff.diff(12) makes the data stationary but as such the predictions won't be interpretable
#as the acf shows clear seasonal spikes when taking diff alone it's not stationary either
#decision for CPI: like reference paper see word document.


#2. simple differencing for non-stationary variables:
#these variables were non-stationary in the adf test but looked stationary: try to difference once and redo adf test:


#Exchange_Rate	Difference (Log)	% Appreciation/Depreciation	Highly Recommended. Levels are useless for forecasting here.
#Wage_change	Caution	Wage Acceleration	Check plot first. If it looks like it fluctuates around a mean, do not difference. Only difference if there is a clear upward/downward trend in the growth rate itself.
#M3_change	Caution	Monetary Impulse	Check plot first. M3 growth is volatile. Differencing it creates "noise on noise," which might confuse the Random Forest
non_s_vars= ['Wage_change','Exchange_Rate_CHF','M3_change']
#take differences of those 
df_non_s = df[non_s_vars].diff().dropna()

#rerun adf for differenced vars:

#initialize list for resutls
adf_results=[]
#loop through all variables of df
for col in df_non_s.columns:
    #all cols shoul dbe float (from preprocessing) but still check to avoid errors
    if pd.api.types.is_numeric_dtype(df_non_s[col]):
        res=run_adf_test(df_non_s[col], col)
        adf_results.append(res)


#to df for nice tabular display
adf_table=pd.DataFrame(adf_results)
#display the table
print("\n Augmented Dickey-Fuller Test Results for Non-Stationary Variables: ")
print(adf_table) 



#-----------------------------------------------
#Result Overview
#-------------------------------------------------



#write a summary of the results seen above
data =[{"Variable": "CPI Targets (Core/Headline)", "Time Series Visual": "Persistent upward trend; acceleration post-2021.", "ACF/PACF Pattern": "Very slow ACF decay; PACF spike at lag 1.",
        "ADF Result (p-value)": "Non-Stationary (0.85 /0.81)", "How to proceed": "Use Log-Growth Rates (Inflation)."},
    {"Variable": "GDP Indexes (ch and eu)", "Time Series Visual": "Smooth, continuous upward trend since 2000.", "ACF/PACF Pattern": "Extremely high ACF persistence (near 1.0).",
        "ADF Result (p-value)": "Non-Stationary (0.95 /0.89)", "How to proceed": "Use Log-Growth Rates."},
    {"Variable": "Sentiment (KOF /Business Conf.)", "Time Series Visual": "Mean-reverting; cyclical 'waves' around a constant.",
        "ACF/PACF Pattern": "ACF drops to zero quickly; some oscillation.", "ADF Result (p-value)": "Stationary (0.000)","How to proceed": "Use Levels (Raw data)."},
    {"Variable": "PPI & Turnover (Retail/Real)", "Time Series Visual": "Distinct upward trend; Retail shows higher noise.", "ACF/PACF Pattern": "ACF remains above confidence interval for 40 lags.",
        "ADF Result (p-value)": "Non-Stationary (0.38 /0.94 /0.83)", "How to proceed": "Use Log-Growth Rates."},
    {"Variable": "Monetary Changes (M1, M2, M3)", "Time Series Visual": "High-frequency volatility; spikes around 2020.",
        "ACF/PACF Pattern": "Faster decay, but high lag count used in ADF.", "ADF Result (p-value)": "Stationary (except M3)", "How to proceed": "Use Levels (verify seasonality)."},
    {"Variable": "Interest Rates (Saron/Mortgages)", "Time Series Visual": "Long periods of zero followed by sudden spikes.",
        "ACF/PACF Pattern": "Highly persistent ACF; PACF lag 1 dominance.", "ADF Result (p-value)": "Stationary (<0.05)",
        "How to proceed": "Use Levels (but watch for breaks)."},
    {"Variable": "Vol_loans", "Time Series Visual": "Clean, linear upward trajectory.", "ACF/PACF Pattern": "Absolute persistence (ACF stays at 1.0).",
        "ADF Result (p-value)": "Non-Stationary (0.99)", "How to proceed": "Use Log-Growth Rates."},
    {"Variable": "Spreads (Fin/EU_Fin)", "Time Series Visual": "Regime-based shifts; spikes during crises (2008/2012).", "ACF/PACF Pattern": "Slow decay; significant autocorrelation.",
        "ADF Result (p-value)": "Borderline (0.15 /0.05)", "How to proceed": "Use First Differences."},
    {"Variable": "Manufacturing_EU", "Time Series Visual": "Structural upward drift with cyclicality.",
        "ACF/PACF Pattern": "Persistent ACF; significant lag-1 PACF.","ADF Result (p-value)": "Non-Stationary (0.79)",
        "How to proceed": "Use Log-Growth Rates."}]

#make df out of it
df_summary =pd.DataFrame(data)
#make nice table out of it for the terminal
table=tabulate(df_summary, headers='keys', tablefmt='grid', showindex=False, maxcolwidths=[20, 25, 25, 20, 20])

#print overview
print("\n Overview Test Results:")
print(table)




#split into train and test set to avoid data leakage

#not adding covid dummies because would be cheating as motivation is to find a model that better copes with the post 2020 period
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#split into one df for core inflation and one for headline inflation

#take log then yearly changes depending on variables maybe also from them






#stabilize the variance




