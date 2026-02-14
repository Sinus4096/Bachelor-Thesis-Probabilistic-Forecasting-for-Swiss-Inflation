import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller
import math





#this script serves as justification for further transformations-> exploratory data analysis of raw data

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

#plot and save
plt.tight_layout(rect=[0, 0, 1, 0.96]) 
save_name=f"Code/Scripts/Plots/02_eda_raw/Time_Series_CPI.png"
plt.savefig(save_name, dpi=300)
#observations:
#there is a clear long-term upward trend in both Core and Headline CPI, showing a steady increase in price levels from 2000 through 2024.
#the Headline CPI exhibits more frequent, jagged fluctuations compared to the Core CPI, likely reflecting the volatile nature of food and energy
#prices which are excluded from the core metric. Additionally, both charts show a noticeable acceleration in the growth rate starting around 2021,
#indicating a period of sharper inflation in recent years.

#to make the shocks from pandemic and energy crisis more visible, plot YoY changes:
#calc yoy changes
df_yoy=df[['Headline_CPI', 'Core_CPI']].pct_change(12)*100
#create figure
fig=plt.figure(figsize=(15, 8), dpi=100)
#define grid
gs=fig.add_gridspec(1, 2, width_ratios=(4, 1), wspace=0.05)

#plot lines of headline and core cpi
ax1= fig.add_subplot(gs[0])
ax1.plot(df_yoy.index, df_yoy['Headline_CPI'], color=headline_color, lw=1.8, label='Headline (YoY)', zorder=3)
ax1.plot(df_yoy.index, df_yoy['Core_CPI'], color=core_color, lw=1.8, label='Core (YoY)', zorder=3)
#add band indicating snb price stability range
ax1.axhspan(0, 2, color='#E8F5E9', alpha=0.4, label='SNB Price Stability Range', zorder=1)
ax1.axhline(0, color='#333333', lw=1, ls='--', zorder=2)
#set titles and labels
ax1.set_title('Inflation Volatility and Tail Risk Periods', fontsize=18, fontweight='bold', loc='left', pad=25)
ax1.set_ylabel('Inflation Rate (YoY %)', fontweight='bold', fontsize=11)
ax1.set_xlabel('Year', fontweight='bold', fontsize=11)
#styling
ax1.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)
ax1.set_ylim(-2.5, 4.5)
#calc kde for both series and plot on the side
ax2 = fig.add_subplot(gs[1], sharey=ax1)
sns.kdeplot(y=df_yoy['Headline_CPI'], ax=ax2, fill=True, color=headline_color, alpha=0.3, linewidth=1.5)
sns.kdeplot(y=df_yoy['Core_CPI'], ax=ax2, fill=True, color=core_color, alpha=0.3, linewidth=1.5)
#labels and styling
ax2.set_xlabel('Frequency Density', fontweight='bold', fontsize=11)
ax2.set_ylabel('')  #minimalism: only have 1 y axis label
ax2.tick_params(axis='y', labelleft=False)
for spine in ['top', 'right', 'left']:  #clean spines
    ax2.spines[spine].set_visible(False)
ax2.grid(axis='x', alpha=0.3)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
save_name=f"Code/Scripts/Plots/02_eda_raw/YoY_CPI_w_distrib.png"
plt.savefig(save_name, dpi=300)




#time series plot for all predictors:

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
        ax.plot(series.index, series.values, label='Raw Level', linewidth=2, color=x_color)
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

    save_name=f"Code/Scripts/Plots/02_eda_raw/Time_Series_features{fig_idx +1}.png"
    plt.savefig(save_name, dpi=300)



#observations as output:
data={
    "variable":["kofbarometer", "unemployment_rate", "oilprices", "gdp_index_ch", "gdp_index_eu", "infl_e_current_year", "infl_e_next_year", 
        "Saron_Rate", "CH_2int", "fin_spread", "EU_fin_spread", "Wage_change", "PPI", "real_turnover", "retail_turnover", "Exchange_Rate_CHF", "variable_mortgages", "Vol_loans", 
        "M1_change", "M2_change", "M3_change", "Manufacturing_EU", "Business_Confidence_EU"],
    "trend":["Stationary/mean-reverting around 100", "Cyclical phases: downward 2010-2020", "Upward 2000-2008, volatile since 2008", "Clear upward trend", 
        "Clear upward trend", "Stationary (0-1%) with structural break 2021", "Stationary (0-1%) with structural break 2021", "Downward 2000-2015, Upward since 2022",
        "Downward 2000-2015, Upward since 2022", "Mean-reverting, no long-term slope","Mean-reverting with large cyclical swings", "High volatility, no long-term trend",
        "Slight upward trend, sharp upward 2021", "Clear long-term upward trend","Clear long-term upward trend", "N/A", "N/A", "Strong upward trend",
        "Mean-reverting, no long-term slope", "Mean-reverting, no long-term slope","Cyclical waves, no clear trend", "Long term upward trend", "Cyclical series"],
    "seasonality": ["None visible", "Jagged, some residual seasonality", "None visible", "None visible", "None visible", "Hard to detect (high volatility)", 
        "None visible", "None visible (policy-controlled)", "None visible","None visible", "None visible", "Spikiness at regular intervals","None visible", "Jagged, possible residual fluctuations", "None visible",
        "N/A", "N/A", "None visible", "None visible", "None visible","None visible", "None visible", "None visible"]}

#create DataFrame
df_summary =pd.DataFrame(data)
#display the table
print(df_summary.to_string(index=False))





#--------------------------------------------
#outlier inspection
#--------------------------------------------
numeric_df= df.select_dtypes(include=['number'])  #select numeric columns
    
#calc IQR
Q1= numeric_df.quantile(0.25)
Q3=numeric_df.quantile(0.75)
IQR=Q3-Q1
#def bounds
lower_bound=Q1- 1.5*IQR
upper_bound =Q3+1.5*IQR
#initialize table with descriptive stats
stats= df.describe().T
#count outliers per column: only extreme outliers beyond 3*IQR
outliers=((numeric_df <(Q1-3.0*IQR))|(numeric_df>(Q3 +3.0*IQR))).sum()
stats['extreme_outliers']= outliers
stats['outlier_pct'] =(outliers/ len(df) *100).round(2).astype(str)+'%'
#want 4 decimals
pd.options.display.float_format ='{:.4f}'.format   
#reorder cols
cols= ['mean', 'std', 'min', '50%', 'max', 'extreme_outliers', 'outlier_pct']
#print
save_name=f"Code/Scripts/Plots/02_eda_raw/stats_and_outlier_insp.csv"
stats[cols].to_csv(save_name) 
print(stats[cols])

#observations:
#outliers are just from economic shocks (see based on min and max)-> no further treatment needed
#probably will need to standardize the data especially for feature importance of qrf and for config with ridge regression

#from now on to prevent data leakage, only use data from 2015 onwards (training data)
df= df.loc[:'2013-07-01']

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
    save_name=f"Code/Scripts/Plots/02_eda_raw/ACF_initial_{chunk_nr}.png"
    plt.savefig(save_name, dpi=300)
    #add 1 to chunknr for next chunk plot
    chunk_nr+=1



#observations as output table:
acf_pacf_data = {
    "variable": ["Core&Headline CPI", "kofbarometer", "unemployment_rate", "oilprices","gdp_index_ch_eu", "infl_e", "Saron_Rate_CH_2int", "fin_spread",
        "EU_fin_spread", "Wage_change", "PPI", "real_turnover","M1_M2", "M3_change", "Manufacturing_EU", "Business_Confidence_EU"],
    "acf_observation": ["Very slow linear decay -> strong trend", "Decays faster, wave like (-> cyclicality)", "Slow decay -> persistent trend, reversal to negative values (cyclicality)", 
        "Slow linear decay -> strong trend", "Extremely slow decay, very strong trend", "Moderately slow decay -> persistent trend-like behavior", 
        "Very slow linear decay -> strong trend", "Slow decay -> short-to-medium term", "Slow decay -> medium term trend-like behavior", "Moderately slow decay -> persistent trend-like behavior", 
        "Extremely slow decay, very strong trend", "Extremely slow decay, very strong trend", "Moderate decay -> persistence in short-to-medium term, mean-reverting", 
        "Decays more slowly -> stronger trend component", "Extremely slow decay, very strong trend","Decays relatively quickly, signature of a cyclical series" ],
    "pacf_observation": ["Dominant spike at lag 1 -> random walk dominance", "Significant spike at lag 1, spike at lag 2 -> AR(2)", "High spike at lag 1, negative spike at lag 2 -> AR(2) structure", 
        "Significant spike at lag 1, spike at lag 2 -> AR(2)", "Massive spike at lag 1 -> random walk with drift", "High spike at lag 1 -> AR structure", 
        "Massive spike at lag 1 -> random walk with drift", "High spike at lag 1 -> AR structure", 
        "Significant spike at lag 1, negative spike at lag 2 -> AR(2)", "Significant spike at lag 1 and 2 -> AR(2) structure", 
        "Significant spike at lag 1 and 2 -> AR(2) structure", "Massive spike at lag 1 -> random walk with drift", "Significant spike at lag 1, negative spike at lag 2 -> AR(2)", 
        "Large spike at lag 1 and smaller spike at lag 2", "Massive spike at lag 1 -> random walk with drift", "Significant spike at lag 1 and 2 -> AR(2) structure" ],
    "seasonality_notes": ["No obvious repeating spikes" ] * 16 #all same obs->*16
}

#df
df_acf_pacf=pd.DataFrame(acf_pacf_data)
#display
print(df_acf_pacf.to_string(index=False))








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
save_name=f"Code/Scripts/Plots/02_eda_raw/ADF_initial.csv"
adf_table.to_csv(save_name) 

#--------------------------------
#decisions for critical cases and general summary
#--------------------------------

#Comparison to the visual analysis and acf/pacf plots:
#No contradicions:
#GDP, PPI, Turnover, Loans, and manufacturing: very high p-values, matches the strong upward trend and slow-decaying acf ->take log differences
#kofbarometer & business_confidence both have p-values of 0-> mean-reverting cyclical indicators with no long-term trend -> keep levels
#M1 & M2: visual inspection showed stationary growth rates and p<0.05 -> keep as they are
#inflation expectations: p<0.01-> stationary-> keep levels
#exchange rate: acf-> long decay and p=0.31 -> take log differences

#Contradictions: 
#interest rates (saron & variable morgages: look non-stationaryas were moving in long steps trending slightly upward but p<0.01-> stationary ->leave at levels
#fin spread (EU&CH): looked stationary, adf-> non stationary-> keep as levels, high-p values in spreads are usually due to structural breaks or long-term persistence, not a true random walk
# -> leave as they are to avoid introducing non-stationary noise in models
#wage change: appeared stationary, adf non stationary, acf->moderate decay, pacf->AR(2)-> keep as levels to avoid introducing non-stationary noise &keep it interpretable
#reason in levels preserves distinction between high-wage and low-wage regimes, allowing the QRF to capture state-dependent threshold effects
#M3: looked similar to M2&M1 but non stationary; trend-like quality-> take differences (not log diff. because already is change in variables-> has negative values)
#reason: persistent cyclical waves containing valuable predictive memory, which the BVAR can effectively model using shrinkage priors

#Borderline cases:
#CH2int: p=0.056-> edge case between random walk and mean-reverting-> keep in levels to not have a highly volatile change in rate










#---------------------------------
#Analysis for seasonality
#---------------------------------
#1. vars with strong trend:
#trend is so dominant that it hides seasonal signa, once calculate the log-growth ratem, trend is removed-> hidden seasonal spikes will become visible.
trending_vars=['Core_CPI', 'Headline_CPI', 'PPI', 'real_turnover', 'retail_turnover', 'Manufacturing_EU', 'Vol_loans','Exchange_Rate_CHF']
#for gdp vars use quarterly diff to match the freq of the data
gdp_vars=['gdp_index_ch', 'gdp_index_eu']
growth_vars= trending_vars +gdp_vars
#take log growth rates and not just differences to make it comparable to target variables
df_growth=np.log(df[trending_vars]).diff().dropna()*100
df_growth[gdp_vars]= np.log(df[gdp_vars]).diff(3).dropna()*100

#start the plotting to see if there is seasonal pattern now:
#set colors nicer than before because those come into thesis
main_color= '#2E4053'  
ci_color= '#AED6F1'
#restrict number of figures to 5
vars_per_fig=4
#def chunknr for title
chunk_nr=1

#loop through variables in chunks to plot acf/pacf
for start_idx in range(0, len(growth_vars), vars_per_fig):
    #def current chunk
    chunk=growth_vars[start_idx: start_idx+vars_per_fig]
    n_vars =len(chunk)    #nr of vars in this chunk (is not 4 if last chunk)
    #create figure 
    fig, axes=plt.subplots(nrows=n_vars, ncols=2, figsize=(10, n_vars*2.2), squeeze=False)
    #loop through vars in chunk
    for i, var_name in enumerate(chunk):
        #get data and drop NA's (before 2001)
        series=df_growth[var_name].dropna()
        #clean name for title
        clean_name =var_name.replace('target_', '').replace('_', ' ').title()
        #acf plot
        plot_acf(series, ax=axes[i, 0], lags=40, color=main_color, vlines_kwargs={"colors": main_color, "linewidth": 1.5}, alpha=0.05, title=f"ACF: {clean_name}")
        #pacf plot
        plot_pacf(series, ax=axes[i, 1], lags=40, color=main_color, vlines_kwargs={"colors": main_color, "linewidth": 1.5},alpha=0.05, title=f"PACF: {clean_name}")
        
        #styling
        for ax in axes[i]:
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.set_ylim(-1.1, 1.1)
                if len(ax.collections) >1:
                    ax.collections[1].set_color(ci_color)
                    ax.collections[1].set_alpha(0.3)
    #figure title for all chunks
    fig.suptitle(f'ACF & PACF Plots of Detrended Variables (Part {chunk_nr})', fontsize=22, fontweight='bold')
    #plot
    plt.tight_layout()
    save_name=f"Code/Scripts/Plots/02_eda_raw/ACF_detrended_{chunk_nr}.png"
    plt.savefig(save_name, dpi=300)
    #add 1 to chunknr for next chunk plot
    chunk_nr+=1
#observations:
#for gdps: look worse (slower decay in acf and ar(2) in pacf) than if do monthly diff but then would have 0's in dataset bc of forward filling-> keep quarterly diff
#all look good except for core and headline cpi: can take both away if do .diff(12).diff(), but then lose interpretability of predictions and probably take out too much info
#->try to use annualized % changes as in ecb working paper:
df_growth=df_growth.drop(['Core_CPI', 'Headline_CPI'], axis=1)
#calculate annualized % changes for both cpi variables as new target variables
horizons=[3, 6, 9, 12]
for h in horizons:
    df_growth[f'target_headline_{h}m'] =(12/h)*(np.log(df['Headline_CPI']).diff(h))* 100
    df_growth[f'target_core_{h}m']=(12/h) *(np.log(df['Core_CPI']).diff(h))*100

#check whether cols exist now:
df_growth.columns
#redo acf check for the cols to check whether look near stationary now:
variables_to_plot= [f'target_headline_{h}m' for h in horizons]+[f'target_core_{h}m' for h in horizons]
#restrict number of figures to 5
vars_per_fig=4
#def chunknr for title
chunk_nr=1

#loop through variables in chunks
for start_idx in range(0, len(variables_to_plot), vars_per_fig):
    
    #def current chunk
    chunk =variables_to_plot[start_idx: start_idx+vars_per_fig]
    n_vars =len(chunk)    #nr of vars in this chunk (is not 4 if last chunk)
    
    # Create fig. squeeze=False ensures 'axes' is ALWAYS a 2D array [row, col]
    fig, axes=plt.subplots(nrows=n_vars, ncols=2, figsize=(10, n_vars*2.2), squeeze=False)
    
    #loop through vars in chunk for acf/pacf plots
    for i, var_name in enumerate(chunk):
        #get data and drop NA's (before 2001)
        series=df_growth[var_name].dropna()
        #clean name for title
        clean_name =var_name.replace('target_', '').replace('_', ' ').title()
        #acf plot
        plot_acf(series, ax=axes[i, 0], lags=40, color=main_color, vlines_kwargs={"colors": main_color, "linewidth": 1.5}, alpha=0.05, title=f"ACF: {clean_name}")
        #pacf plot
        plot_pacf(series, ax=axes[i, 1], lags=40, color=main_color, vlines_kwargs={"colors": main_color, "linewidth": 1.5},alpha=0.05, title=f"PACF: {clean_name}")
        
        #styling
        for ax in axes[i]:
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.set_ylim(-1.1, 1.1)
                if len(ax.collections) >1:
                    ax.collections[1].set_color(ci_color)
                    ax.collections[1].set_alpha(0.3)
    #figure title for all chunks
    fig.suptitle(f'ACF & PACF Plots of Annualized Targets (Part {chunk_nr})', fontsize=22, fontweight='bold')
    #plot
    plt.tight_layout()
    save_name=f"Code/Scripts/Plots/02_eda_raw/ACF_annualized_CPI{chunk_nr}.png"
    plt.savefig(save_name, dpi=300)
    chunk_nr+=1

#see whether lags after lag 2 in pacf plot are from seasonality rather than trend-> difference (won't do to preprocess but needed
#to determine how many lags to include in models):
for h in horizons:
    df_growth[f'target_headline_{h}m_diff'] =df_growth[f'target_headline_{h}m'].diff(12)
    df_growth[f'target_core_{h}m_diff'] =df_growth[f'target_core_{h}m'].diff(12)
variables_to_plot= [f'target_headline_{h}m_diff' for h in horizons]+[f'target_core_{h}m_diff' for h in horizons]
#reset chunk nr
chunk_nr=1
#plot acf/pacf
for start_idx in range(0, len(variables_to_plot), vars_per_fig):
    
    #def current chunk
    chunk =variables_to_plot[start_idx: start_idx+vars_per_fig]
    n_vars =len(chunk)    #nr of vars in this chunk (is not 4 if last chunk)
    
    # Create fig. squeeze=False ensures 'axes' is ALWAYS a 2D array [row, col]
    fig, axes=plt.subplots(nrows=n_vars, ncols=2, figsize=(10, n_vars*2.2), squeeze=False)
    
    #loop through vars in chunk for acf/pacf plots
    for i, var_name in enumerate(chunk):
        #get data and drop NA's (before 2001)
        series=df_growth[var_name].dropna()
        #clean name for title
        clean_name =var_name.replace('target_', '').replace('_', ' ').title()
        #acf plot
        plot_acf(series, ax=axes[i, 0], lags=40, color=main_color, vlines_kwargs={"colors": main_color, "linewidth": 1.5}, alpha=0.05, title=f"ACF: {clean_name}")
        #pacf plot
        plot_pacf(series, ax=axes[i, 1], lags=40, color=main_color, vlines_kwargs={"colors": main_color, "linewidth": 1.5},alpha=0.05, title=f"PACF: {clean_name}")
        
        #styling
        for ax in axes[i]:
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.set_ylim(-1.1, 1.1)
                if len(ax.collections) >1:
                    ax.collections[1].set_color(ci_color)
                    ax.collections[1].set_alpha(0.3)
    #figure title for all chunks
    fig.suptitle(f'ACF & PACF Plots of Annualized and Differenced Targets (Part {chunk_nr})', fontsize=22, fontweight='bold')
    #plot
    plt.tight_layout()
    save_name=f"Code/Scripts/Plots/02_eda_raw/ACF_annualized_differenced_CPI{chunk_nr}.png"
    plt.savefig(save_name, dpi=300)
    chunk_nr+=1
#acf:data still has long-term cyclicality (eg business cycles)
#pacf 3m: lag at 4, 12m lat at 12 still significant, probably due to how annualized rates are calculated
#for 6m and 9m only lags also around 10 still significant
#decision: will not remove them as fixing them will cause the model to rather overfit to the window's construction rather 
#than the economy.
#still see high persistence especially when h increases, still choose this approach: see thesis chapter 3.1

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
save_name=f"Code/Scripts/Plots/02_eda_raw/ADF_final.csv"
adf_table.to_csv(save_name) 
#remark:with experimetning, we can see that diff.diff(12) makes the data stationary but as such the predictions won't be interpretable
#as the acf shows clear seasonal spikes when taking diff alone it's not stationary either
#decision for CPI: like reference paper see word document.



#-----------------------------------
#yoy config
df_growth[f'headline_direct'] =(np.log(df['Headline_CPI']).diff(12))* 100
df_growth[f'core_direct']=(np.log(df['Core_CPI']).diff(12))* 100

#do acf check for the cols to check whether look near stationary now:
variables_to_plot= [f'headline_direct']+[f'core_direct']
#restrict number of figures to 4 (only have 2 but want to keep code consistent)
vars_per_fig=4
#def chunknr for title
chunk_nr=1

#loop through variables in chunks
for start_idx in range(0, len(variables_to_plot), vars_per_fig):
    
    #def current chunk
    chunk =variables_to_plot[start_idx: start_idx+vars_per_fig]
    n_vars =len(chunk)    #nr of vars in this chunk (is not 4 if last chunk)
    
    # Create fig. squeeze=False ensures 'axes' is ALWAYS a 2D array [row, col]
    fig, axes=plt.subplots(nrows=n_vars, ncols=2, figsize=(10, n_vars*2.2), squeeze=False)
    
    #loop through vars in chunk for acf/pacf plots
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
    fig.suptitle(f'ACF & PACF Plots of YoY-% Change Targets (Part {chunk_nr})', fontsize=22, fontweight='bold')
    #plot
    plt.tight_layout()
    plt.show()
    chunk_nr+=1

#can also take the same approach for lags and will leave the sine and cosine terms in 