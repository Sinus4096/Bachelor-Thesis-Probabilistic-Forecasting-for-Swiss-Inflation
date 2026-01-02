import requests
import pandas as pd
from pyjstat import pyjstat
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import os
import re
warnings.filterwarnings('ignore')

#define path to load the datasets
script_dir =Path(__file__).resolve().parent
BASE_PATH =script_dir.parent.parent/'Code'/'Data'/'Raw_Data'

#----------------------
#define all function to load the raw data

def load_cpi_data(file_path):
    """load CPI year-over-year changes"""    
    #read 'VAR_m-12' sheet 
    cpi_data_wide =pd.read_excel(file_path/'Inflation.xlsx', sheet_name='VAR_m-12', header=3)
    #define which rows to extract and filtered df for those
    target_rows=['Total', '    Kerninflation 1']
    filtered_data=cpi_data_wide[cpi_data_wide['Position_D'].isin(target_rows)].copy()
    #remove identical rows and convert to long
    filtered_data=filtered_data.drop_duplicates()
    data_long =filtered_data.melt(id_vars=['Position_D'], var_name='Date', value_name='Inflation')

    #skip the first 26 rows, as Kerninflation 1 only starts after them
    data_long =data_long.iloc[26:]   
    #date to datetime obj 
    data_long['Date'] =pd.to_datetime(data_long['Date'])
    
    final_cpi_df=data_long.pivot(index='Date', columns='Position_D', values='Inflation').reset_index()  #reshape 
    final_cpi_df.columns=['Date', 'Core_CPI', 'Headline_CPI']   #rename
    #'Date' as index and ensure every month is present (filling missing months with NaN)
    return final_cpi_df.set_index('Date').resample('MS').asfreq()


def load_kof_barometer(file_path):
    """load KOF data"""
    KOF_barometer =pd.read_excel(file_path/'KOF_economic_barometer.xlsx')
    #date to datetime obj and drop date col
    KOF_barometer['Date']=pd.to_datetime(KOF_barometer['date'])
    KOF_barometer =KOF_barometer.drop('date', axis=1)
    #return with date as index (to match cpi data for joining)
    return KOF_barometer.set_index('Date')


def load_unemployment(file_path):
    """load and process unemployment rate
    """
    u_rate =pd.read_excel(file_path /'Arbeitslosenquote.xlsx', sheet_name=None, header=4)
    #initialize list to store the data
    data_list=[]
    
    #looop through dictionary of dfs (year=sheet name, df=the data for that year)
    for year, df in u_rate.items():
        #iterate row by row
        for i in range(len(df)):
            #look at value in 1. column (where names are) 
            cell_value=df.iloc[i, 0]
            #check row label is total
            if pd.notna(cell_value) and str(cell_value).strip().lower()=='total':
                #grab entire row of total data
                total_row = df.iloc[i]
                
                #loop through columns of this row
                for col_idx in range(1, len(df.columns)):
                    col =df.columns[col_idx]      #column header year-month
                    value =total_row.iloc[col_idx] #unemployment percentage
                    
                    #ensure not picking up empty data
                    if pd.notna(value) and pd.notna(col) and col != 'Unnamed: 0':
                        date_str=str(col).strip()
                        #convert to datetime object adn append
                        date = pd.to_datetime(date_str, format='%Y-%m', errors='coerce')                        
                        #append if date conversion worked
                        if pd.notna(date):
                            data_list.append({'date': date, 'total': value})

                #total row is processed for sheet-> next sheet
                break
    
    #convert into df and sort by date
    Unemployed=pd.DataFrame(data_list)    
    Unemployed=Unemployed.sort_values('date').reset_index(drop=True)
    
    #date as index
    Unemployed.set_index('date', inplace=True)
    
    #rename
    Unemployed.columns =['unemployment_rate']    
    #fill missing months with NaN
    return Unemployed.resample('MS').asfreq()





def load_oil_prices(file_path):
    """load oil prices
    """
    oil_prices =pd.read_excel(file_path/'FRED_Crude_Oil_Prices.xlsx', sheet_name='Monthly')
    oil_prices['Date'] =pd.to_datetime(oil_prices['observation_date'])  #datetime obj
    oil_prices = oil_prices.drop('observation_date', axis=1)    #drop initial date column
    oil_prices.set_index('Date', inplace=True)  #set index
    oil_prices.columns=['oilprices']    #rename
    return oil_prices.resample('MS').asfreq()


def load_gdp_ch(file_path):
    """
    load quarterly (-> ffill) swiss GDP"""
    gdp=pd.read_excel(file_path/'gdp.xlsx', sheet_name='real_q', header=10)
    #select year, quarter and percentage change cols
    gdp = gdp.iloc[:, [0, 1, 3]]
    gdp.columns=['Year', 'Quarter', 'real_gdp_growth']    #rename
    #crate the date given year and quarter
    gdp['Date'] = pd.to_datetime(gdp['Year'].astype(str) + 'Q' + gdp['Quarter'].astype(str))
    gdp.set_index('Date', inplace=True)
    #fill up missing values (due to monthly data); use forward fill to avoid data leakage
    return gdp[['real_gdp_growth']].resample('MS').ffill()



def load_gdp_eu(file_path):
    """load EU GDP (quarterly)"""
    gdp2 =pd.read_excel(file_path/'gdp_EU.xlsx', sheet_name='Quarterly')
    #calculate the percentage change of gdp?
    gdp2['gdp_EU_growth']=gdp2['CLVMNACSCAB1GQEU272020'].pct_change()
    gdp2=gdp2.drop('CLVMNACSCAB1GQEU272020', axis=1)  #drop raw rates
    gdp2.columns =['Date', 'gdp_EU_growth']    #rename cols
    gdp2['Date']=pd.to_datetime(gdp2['Date']) #date to datetime object
    #set date as index and fill up missing months with forward fill as done with swiss gdp growth
    return gdp2.set_index('Date')[['gdp_EU_growth']].resample('MS').ffill()


def load_inflation_expectations(file_path):
    """load inflation expectations (quarterly)
    """
    infl_exp=pd.read_excel(file_path/'KOF_consensus_forecast.xlsx')
    #choose cols and rename them
    infl_exp =infl_exp[['date', 'ch.kof.consensus.q_qn_prices_cy.mean', 'ch.kof.consensus.q_qn_prices_ny.mean']]
    infl_exp.columns=['Date', 'infl_e_current_year', 'infl_e_next_year']
    infl_exp['Date']=pd.to_datetime(infl_exp['Date'])   #to datetime object
    infl_exp.set_index('Date', inplace=True)        #set date as index

    #resample to month start and forward fill to avoid data leakage and have monthly data
    infl_exp_monthly = infl_exp.resample('MS').ffill()
    return infl_exp_monthly




def load_interest_rates(file_path):
    """load and process interest rates and create financial spread (daily data)
    """
    #short term interest rates
    st_libor=pd.read_excel(file_path/'SNB_3-month_CHF_Libor_shortterm.xlsx', header=15)
    st_libor.columns = ['Date', 'Saron_Rate', 'Call_money_rate', 'Governmental_claims', '3m_CHF_Libor']     #select cols
    st_libor['Date']=pd.to_datetime(st_libor['Date'])   #to datetime object
    st_libor.set_index('Date', inplace=True)    #set index
    
    #long term interest rates
    lr_interest=pd.read_excel(file_path/'SNB_Spot_interest_rates_longterm.xlsx', header=15)
    lr_interest.columns =['Date', 'EU_2int', 'EU_10int', 'CH_2int', 'CH_10int']
    lr_interest['Date'] =pd.to_datetime(lr_interest['Date'])
    lr_interest.set_index('Date', inplace=True) #set index

    #merge the two dfs    
    df =pd.concat([st_libor, lr_interest], axis=1, join='outer')
    #take monthly mean to smooth daily volatility (no data leakage as will devide to train and test monthly) 
    df =df.resample('MS').mean()
    #calculate the financial spread defined as longterm int - short term int
    df['fin_spread'] =df['CH_10int']-df['Saron_Rate']
    return df[['Saron_Rate', 'CH_2int', 'fin_spread']]



def load_eu_interest(file_path):
    """load EU interest spread, same as for CH"""
    #long term interest rates
    EU_interest=pd.read_csv(file_path/'Euro_Area_Interest.csv')
    EU_interest.columns =['Date', 'EU_int'] #select cols
    EU_interest['Date']=pd.to_datetime(EU_interest['Date'])     #as datetime obj
    EU_interest.set_index('Date', inplace=True) #sest index
    
    #short term interest rates
    EU_short_term_int =pd.read_csv(file_path/'Euribor_Euro_area.csv')
    EU_short_term_int.columns=['Date', 'Time', 'short_term_int_EU']
    EU_short_term_int['Date'] =pd.to_datetime(EU_short_term_int['Date'])
    #dates to the first of the month
    EU_short_term_int['Date'] = EU_short_term_int['Date'].dt.to_period('M').dt.to_timestamp()
    EU_short_term_int.set_index('Date', inplace=True)   

    #merge and calculate the financial spread
    df=pd.merge(EU_interest, EU_short_term_int, left_index=True, right_index=True, how='left')
    df['EU_fin_spread'] =df.iloc[:,0]-df['short_term_int_EU']
    return df[['EU_fin_spread']].resample('MS').mean()



def load_wages(file_path):
    """load wage data (quarterly)
    """
    
    df =pd.read_excel(file_path/'quarterly_nominal_wage_development.xlsx', header=3)
    #initialize list
    tidy_data=[]
    #find row indices where first column contains phrase "Annual variation..."
    data_row_indices=df[df.iloc[:, 0].astype(str).str.contains('Annual variation of nominal wages', na=False)].index
    #map roman numerals (I, II, III, IV) used for quarters
    quarter_to_month_map ={'I': 1, 'II': 4, 'III': 7, 'IV': 10}
    #iterate through every instance where annual variation row was found
    for idx in data_row_indices:
        #step back 4 rows from anchor to find year, then forward fill the year for the other quarters of the same year (i.e. same row)
        year_row_filled =df.iloc[idx - 4].ffill()
        quarter_row =df.iloc[idx - 3]   #find quarter values
        value_row =df.iloc[idx]     #value: anchor itself
        
        #iterate through each column (skipping the label column at index 0)
        for col_idx in range(1, len(df.columns)):
            #get dates and raw value
            year = year_row_filled.iloc[col_idx]
            quarter_raw = quarter_row.iloc[col_idx]
            raw_value = value_row.iloc[col_idx]
            #some values have 'a' if they're uncertain-> extract numerical part or forward fill as we only have short gaps
            if pd.notna(year) and pd.notna(quarter_raw) and pd.notna(raw_value):
                    #map quarters to months
                    quarter_clean=str(quarter_raw).strip().upper()
                    month =quarter_to_month_map.get(quarter_clean)

                    #eExtract numerical part using regex
                    raw_value_str=str(raw_value).strip()
                    match =re.match(r'^(-?\d+(\.\d+)?)', raw_value_str)
                    
                    wage_change_num =float(match.group(1)) if match else None
                    #append tidied data to dict
                    if wage_change_num is not None:
                        tidy_data.append({'Year': int(year),'Month': month,'Wage_change': wage_change_num})
    wages_df = pd.DataFrame(tidy_data)  #to df
    #initialize to first of month
    wages_df['Day']=1
    #define date
    wages_df['Date'] =pd.to_datetime(wages_df[['Year', 'Month', 'Day']], errors='coerce')  
    #set Date as index, sort chronologically, and ensure monthly frequency  
    wages_df =wages_df[['Date', 'Wage_change']].set_index('Date').sort_index()  
    #make sure start at month start and forward fill to have monthly data
    return wages_df.resample('MS').ffill()




def load_turnover_and_ppi(file_path):
    """load industrial turnover and PPI (quarterly data)
    """
    #URL for the Swiss Federal Statistical Office PX-Web API
    url='https://www.pxweb.bfs.admin.ch/api/v1/en/px-x-0603010000_102/px-x-0603010000_102.px'
    #JSON dictionary defining filters
    json_query={"query": [{"code": "Bereinigung", "selection": {"filter": "item", "values": ["sa"]}},{"code": "Indizes / Veränderungsraten", "selection": {"filter": "item", "values": ["ind"]}},
            {"code": "Variable", "selection": {"filter": "item", "values": ["utot"]}},{"code": "Branche", "selection": {"filter": "item", "values": ["C"]}}],
        "response": {"format": "json-stat"}}
    #sends request to BFS server and convert specialized 'json-stat' format into df
    response=requests.post(url, json=json_query)
    turnover=pyjstat.from_json_stat(response.json())[0]
    turnover=turnover[['Quarter', 'value']]     #select colst
    #convert quarterly strings (e.g., "2023Q1") into date strings (e.g., "2023-01-01")
    turnover['Date']=turnover['Quarter'].str.replace('Q1', '-01-01').str.replace('Q2', '-04-01').str.replace('Q3', '-07-01').str.replace('Q4', '-10-01')
    turnover['Date']=pd.to_datetime(turnover['Date']) #to datetime object
    turnover =turnover.set_index('Date')       #set date to index
    #forward fill to expand quarterly turnover to monthly data
    turnover =turnover.resample('MS').ffill()

    #load PPI
    PPI =pd.read_excel(file_path/'PPI.xlsx', sheet_name='INDEX_m', header=6, skiprows=range(7, 80))
    PPI= PPI[0:1195]
    #select the date and the specific index column
    PPI=PPI[['Datum', 'Dez 2020 = 100']]
    PPI['Datum'] =pd.to_datetime(PPI['Datum'])
    PPI.columns =['Date', 'PPI']
    PPI=PPI.set_index('Date')
    
    #join the two dfs
    turnover =turnover.merge(PPI, left_index=True, right_index=True, how='left')
    #calc real turnover by dividing nominal turnover by Price Index-> removes effect of price changes (inflation) 
    turnover['real_turnover']=(turnover['value'] / turnover['PPI'])*100
    turnover =turnover[['PPI', 'real_turnover']]
    return turnover.resample('MS').asfreq()



def load_retail():
    """load retail turnover
    """

    #def url and JSON dictionary defining filters
    url2='https://www.pxweb.bfs.admin.ch/api/v1/en/px-x-0603020000_101/px-x-0603020000_101.px'
    json_query2 = {"query": [{"code": "Bereinigung", "selection": {"filter": "item", "values": ["sa"]}},{"code": "Indizes / Veränderungen", "selection": {"filter": "item", "values": ["ind"]}},
            {"code": "Nominal / Real", "selection": {"filter": "item", "values": ["r"]}},{"code": "Branche / Warengruppe", "selection": {"filter": "item", "values": ["47"]}}
        ],
        "response": {"format": "json-stat"}}
    #sends request to BFS server and convert specialized 'json-stat' format into df   
    response2 =requests.post(url2, json=json_query2)
    retail =pyjstat.from_json_stat(response2.json())[0]     #into list of dfs
    retail =retail[['Month', 'value']]      #select cols
    #ensure Month column is treated as string
    retail['Month']=retail['Month'].astype(str)
    #convert BFS month format (e.g., "2023M05") to standard format ("2023-05")
    retail['Date']=retail['Month'].str.replace('M', '-')
    retail['Date'] =pd.to_datetime(retail['Date'])     #to datetime object
    retail =retail[['value', 'Date']]       #select cols
    retail.columns=['retail_turnover', 'Date']      #rename
    return retail.set_index('Date').resample('MS').asfreq()


def load_exchange_rate(file_path):
    """Load exchange rates
    """
    Exchange=pd.read_excel(file_path/'SNB_Exchange_rates.xlsx', header=15, sheet_name=0)
    Exchange =Exchange.iloc[:, [0, 1]] #first two cols only
    Exchange.columns =['Date', 'Exchange_Rate_CHF'] #rename
    Exchange['Date']=pd.to_datetime(Exchange['Date'])   #to datetime object
    return Exchange.set_index('Date').resample('MS').asfreq()


def load_mortgages(file_path):
    """load mortgage data
    """
    mortgages =pd.read_csv(file_path/'Variable_mortgages.csv', header=2, sep=';')
    mortgages =mortgages[['Date', 'Value']]     #select
    mortgages['Date'] =pd.to_datetime(mortgages['Date'])        #to datetime obj
    mortgages.columns =['Date', 'variable_mortgages']   #rename
    return mortgages.set_index('Date').resample('MS').asfreq()

def load_volume_loans(file_path):
    """Load volume of loans
    """
    Vol_loans =pd.read_csv(file_path/'volume_loans.csv', header=2, sep=';')
    Vol_loans=Vol_loans[['Date', 'Value']]
    Vol_loans['Date']=pd.to_datetime(Vol_loans['Date'])
    Vol_loans.columns=['Date', 'Vol_loans']
    return Vol_loans.set_index('Date').resample('MS').asfreq()

def load_money_supply(file_path):
    """Load money supply"""
    Money_sup=pd.read_csv(file_path/'Monetary_aggregate_change.csv', header=2, sep=';')
    Money_sup['Date']=pd.to_datetime(Money_sup['Date'])
    #reshape data: take labels in 'D1' col (M1, M2, M3) and turn into individual columns
    Money_sup=  Money_sup.pivot(index='Date', columns='D1', values='Value')
    Money_sup.columns=['M1_change', 'M2_change', 'M3_change']
    return Money_sup.resample('MS').asfreq()

def load_manufacturing_eu(file_path):
    """Load EU manufacturing data
    """
    Manuf =pd.read_csv(file_path/'Manufacturing_for_EU.csv')
    Manuf['observation_date'] =pd.to_datetime(Manuf['observation_date'])
    Manuf.columns =['Date', 'Manufacturing_EU']
    return Manuf.set_index('Date').resample('MS').asfreq()

def load_business_confidence_eu(file_path):
    """
    load EU business confidence"""
    B_Conf =pd.read_csv(file_path /'Business_Confidence_EU.csv')
    B_Conf['observation_date'] =pd.to_datetime(B_Conf['observation_date'])
    B_Conf.columns =['Date', 'Business_Confidence_EU']
    return B_Conf.set_index('Date').resample('MS').asfreq()

#-----------------------------------

#load all datasets through the above defined functions using pipeline logic

def load_all_data_parallel(file_path):
    """
    load all data sources in parallel for efficiency"""
    
    #define dictionary mapping names to lambda functions
    file_loaders ={'cpi': lambda: load_cpi_data(file_path), 'kof_barometer': lambda: load_kof_barometer(file_path), 'unemployment': lambda: load_unemployment(file_path),
        'oil_prices': lambda: load_oil_prices(file_path), 'gdp_ch': lambda: load_gdp_ch(file_path), 'gdp_eu': lambda: load_gdp_eu(file_path), 'inflation_exp': lambda: load_inflation_expectations(file_path),
        'interest': lambda: load_interest_rates(file_path),'eu_interest': lambda: load_eu_interest(file_path),'wages': lambda: load_wages(file_path),
        'turnover_ppi': lambda: load_turnover_and_ppi(file_path), 'exchange': lambda: load_exchange_rate(file_path),'mortgages': lambda: load_mortgages(file_path),
        'vol_loans': lambda: load_volume_loans(file_path), 'money_supply': lambda: load_money_supply(file_path),
        'manufacturing_eu': lambda: load_manufacturing_eu(file_path), 'business_conf_eu': lambda: load_business_confidence_eu(file_path)}
    #initialize dictionary 
    results ={}
    
    #create thread pool with 8 workers 
    with ThreadPoolExecutor(max_workers=8) as executor:
        #submit all tasks to executor
        future_to_name ={executor.submit(loader): name for name, loader in file_loaders.items()}
        
        #when each task finishes-> get result
        for future in as_completed(future_to_name):
            name =future_to_name[future]
            #store df in dictionary
            results[name] =future.result()
    
    #load API data separately
    results['retail']=load_retail()
    
    return results

def merge_all_data(data_dict):
    """Merge all loaded datasets into single data_before_split DataFrame"""
    print("\n=== Merging all datasets ===")
    
    #start with CPI data
    data_before_split =data_dict['cpi'].copy()
    
    #define merge order 
    merge_order =['kof_barometer', 'unemployment', 'oil_prices', 'gdp_ch', 'gdp_eu','inflation_exp', 'interest', 'eu_interest', 'wages', 'turnover_ppi',
        'retail', 'exchange', 'mortgages', 'vol_loans', 'money_supply', 'manufacturing_eu', 'business_conf_eu']
    
    #merge and verify all get merged
    for key in merge_order:
        if data_dict[key] is not None:
            data_before_split =pd.merge(data_before_split, data_dict[key], left_index=True, right_index=True, how='left')
            print(f"Merged {key}")
        else:
            print(f"Skipped {key} (not loaded)")
    #print shape as 2. check
    print(f"\nshape: {data_before_split.shape}")
    return data_before_split

#execution
if __name__ =="__main__":
    #load all data in parallel
    data_dict =load_all_data_parallel(BASE_PATH)
    
    # Merge into final dataset
    data_before_split =merge_all_data(data_dict)
    
    #save as csv
    CODE_DIR=Path(__file__).parent.parent
    output_path =CODE_DIR /"Data"/"Cleaned_Data"
    output_path.mkdir(parents=True, exist_ok=True) 
    output_file =output_path /'data_before_split.csv'
    data_before_split.to_csv(output_file, index=True)

#look at part where all values exist:
start_date ='2000-05-01'
end_date ='2025-04-01'
data_before_split = data_before_split.loc[start_date:end_date]
data_before_split.info()

#Core_CPI is an object-> coerce to float
data_before_split['Core_CPI'] = pd.to_numeric(data_before_split['Core_CPI'], errors='coerce') 
data_before_split['Headline_CPI'] = pd.to_numeric(data_before_split['Headline_CPI'], errors='coerce') 
#check for NaNs
nans_per_column = data_before_split.isna().sum()
print(nans_per_column)
#check which years are missing
print(data_before_split[data_before_split['infl_e_current_year'].isna()])
#only in year 2000-> can ignore as taking yearly change so will use the observation in year 2000 eitherway
