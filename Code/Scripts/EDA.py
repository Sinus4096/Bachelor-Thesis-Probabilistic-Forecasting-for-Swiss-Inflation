import requests
import pandas as pd
from pyjstat import pyjstat
import numpy as np

#CPI data: VAr m 12 because it represents the year-over-year inflation rate
file_path = 'C:/Users/Sina/Documents/HSG/Bachelor_Thesis/data' 
cpi_data_wide = pd.read_excel(file_path + '/Inflation.xlsx', sheet_name='VAR_m-12', header=3)
#Filter for Headline and Core inflation
target_rows = ['Total', '    Kerninflation 1']
filtered_data = cpi_data_wide[cpi_data_wide['Position_D'].isin(target_rows)].copy()
#have the row 'Total' twice because there were 2 different ways used to measure 
filtered_data = filtered_data.drop_duplicates()
filtered_data
#reshape the dataframe: from wide to long
id_columns = ['Position_D']
data_long = filtered_data.melt(
    id_vars=id_columns, 
    var_name='Date', 
    value_name='Inflation'
)
data_long.head()
#'Date' column to a proper datetime format.
data_long = data_long.iloc[26:]
data_long['Date'] = pd.to_datetime(data_long['Date'])
data_long
# Pivot the table to have Headline and Core CPI as separate columns
final_cpi_df = data_long.pivot(
    index='Date',
    columns='Position_D',
    values='Inflation'
).reset_index()
final_cpi_df.columns= ['Date', 'Core_CPI', 'Headline_CPI']
QRF_data = final_cpi_df.copy()

# Set the date as the index
QRF_data.set_index('Date', inplace=True)



#KOF Economic barometer
KOF_barometer = pd.read_excel(file_path +'/KOF_economic_barometer.xlsx')
KOF_barometer['Date'] = pd.to_datetime(KOF_barometer['date'])
KOF_barometer=KOF_barometer.drop('date', axis=1)
KOF_barometer.set_index('Date', inplace=True)
KOF_barometer
QRF_data
QRF_data= pd.merge(QRF_data, KOF_barometer, left_index=True, right_index=True)

#Unemployment rate
u_rate = pd.read_excel(file_path +'/Arbeitslosenquote.xlsx', sheet_name=None, header=4)
u_rate
data_list = []

#Analyze dataset
first_sheet_name = list(u_rate.keys())[0]
first_df = u_rate[first_sheet_name]
print(f"First sheet: {first_sheet_name}")
print(f"Shape: {first_df.shape}")
print(f"Columns: {list(first_df.columns)}")
print("\nFirst 10 rows, first column:")

# go through all sheets and look for "total" column in first rows
for year, df in u_rate.items():
    for i in range(len(df)):
        cell_value = df.iloc[i, 0]
        if pd.notna(cell_value) and str(cell_value).strip().lower() == 'total':
            total_row = df.iloc[i]
            # Get all columns from index 2 onwards and create a date string from the column names
            for col in df.columns[2:]:
                value = total_row[col]
                if pd.notna(value):
                    date_str = f"{col}"
                    data_list.append({'date': date_str, 'total': value})
            break

print(f"\nTotal entries collected: {len(data_list)}")
print(data_list)
# Create DataFrame
Unemployed = pd.DataFrame(data_list)
Unemployed['date'] = pd.to_datetime(Unemployed['date'])
Unemployed = Unemployed.sort_values('date').reset_index(drop=True)
print(Unemployed)
Unemployed.set_index('date', inplace=True)
Unemployed.columns= ['unemployment_rate']
QRF_data= pd.merge(QRF_data, Unemployed, left_index=True, right_index=True)



#Fred oil prices
oil_prices= pd.read_excel(file_path+ '/FRED_Crude_Oil_Prices.xlsx', sheet_name= 'Monthly')
oil_prices['Date'] = pd.to_datetime(oil_prices['observation_date'])
oil_prices=oil_prices.drop('observation_date', axis=1)
oil_prices.set_index('Date', inplace=True)
oil_prices.columns=['oilprices']
oil_prices
QRF_data= pd.merge(QRF_data, oil_prices, left_index=True, right_index=True)
#change to year to year price changes


#GDP
gdp = pd.read_excel(file_path + '/gdp.xlsx', sheet_name='real_q', header=10)
gdp= gdp.iloc[:, [0, 1, 3]]
gdp.columns = ['Year', 'Quarter', 'real gdp growth']
#columns Year and Quarter to a single date column
gdp['Date'] = pd.to_datetime(gdp['Year'].astype(str) + 'Q' + gdp['Quarter'].astype(str))
gdp= gdp.drop(['Quarter', 'Year'], axis=1)
gdp.set_index('Date', inplace=True)

#forward fill to get from quarterly to monthly data and set to start of month
gdp_CH = gdp.resample('MS').ffill()
gdp_CH
QRF_data= pd.merge(QRF_data, gdp_CH, left_index=True, right_index=True)


#EU GDP
gdp2 = pd.read_excel(file_path + '/gdp_EU.xlsx', sheet_name='Quarterly')
#calculate pct growth rate
gdp2['gdp_EU_growth'] = gdp2['CLVMNACSCAB1GQEU272020'].pct_change() 
gdp2=gdp2.drop('CLVMNACSCAB1GQEU272020', axis=1)
gdp2.columns = ['Date', 'gdp_EU_growth']
gdp2.set_index('Date', inplace=True)
#forward fill
gdp_EU= gdp2.resample('MS').ffill()
gdp_EU
QRF_data= pd.merge(QRF_data, gdp_EU, left_index=True, right_index=True)



#KOF inflation expectations
infl_exp= pd.read_excel(file_path +'/KOF_consensus_forecast.xlsx')
infl_exp= infl_exp[['date', 'ch.kof.consensus.q_qn_prices_cy.mean', 'ch.kof.consensus.q_qn_prices_ny.mean']]
infl_exp.columns = ['Date', 'infl_e_current_year', 'infl_e_next_year']
infl_exp['Date'] = pd.to_datetime(infl_exp['Date'])
infl_exp.set_index('Date', inplace=True)
#backward fill because KOF sends out a questionnaire at beginning of quarter (January etc.), but want startning date 01.01.2001-> define range
monthly_range = pd.date_range(start='2001-01-01', end='2025-06-01', freq='MS')
infl_exp = infl_exp.reindex(monthly_range).fillna(method='bfill')
infl_exp
QRF_data= pd.merge(QRF_data, infl_exp, left_index=True, right_index=True)


#Financial spread
#CHF libor shortterm
st_libor= pd.read_excel(file_path+'/SNB_3-month_CHF_Libor_shortterm.xlsx', header=15)
st_libor.columns= ['Date', 'Saron_Rate', 'Call_money_rate', 'Governmental_claims', '3m_CHF_Libor']
st_libor['Date'] = pd.to_datetime(st_libor['Date'])
st_libor.set_index('Date', inplace=True)
st_libor
#SNB long term interest rates
lr_interest= pd.read_excel(file_path+'/SNB_Spot_interest_rates_longterm.xlsx', header=15)
lr_interest.columns= ['Date', 'EU_2int', 'EU_10int', 'CH_2int', 'CH_10int']
lr_interest['Date'] = pd.to_datetime(lr_interest['Date'])
lr_interest.set_index('Date', inplace=True)
#convert daily data to monthly average
lr_interest = lr_interest.resample('MS').mean()
lr_interest
#join the two dataframes
interest = pd.concat([st_libor, lr_interest], axis=1, join='outer')
interest
#create variable for the financial spread
interest['fin_spread']= interest['CH_10int']-interest['Saron_Rate']
#drop redundant variables
interest=interest[['Saron_Rate', 'CH_2int', 'fin_spread']]
QRF_data= pd.merge(QRF_data, interest, left_index=True, right_index=True)
QRF_data
#EU interest rates for financial spread in one of most important trading partners (EU area)
EU_interest= pd.read_csv(file_path + '/Euro_Area_Interest.csv')
EU_interest.columns= ['Date', 'EU_int']
EU_interest['Date'] = pd.to_datetime(EU_interest['Date'])

EU_short_term_int= pd.read_csv(file_path +'/Euribor_Euro_area.csv')
EU_short_term_int.columns=['Date', 'Time', 'short_term_int_EU']
EU_short_term_int['Date']= pd.to_datetime(EU_short_term_int['Date'])
#need to change date to first of month
EU_short_term_int['Date'] = EU_short_term_int['Date'].dt.to_period('M').dt.to_timestamp()
EU_short_term_int.set_index('Date', inplace=True)
EU_interest.set_index('Date', inplace=True)
EU_short_term_int
#merge the two dataframes
EU_interest= pd.merge(EU_interest, EU_short_term_int, left_index=True, right_index=True, how='left')
#calculate financial spread
EU_interest['EU_fin_spread']=  EU_interest['EU_int'] - EU_interest['short_term_int_EU']
EU_interest= EU_interest['EU_fin_spread']
QRF_data= pd.merge(QRF_data, EU_interest, left_index=True, right_index=True)
QRF_data





#Quarterly estimate of nominal wage development" as a proxy for the Swiss wage index 
df= pd.read_excel( file_path+'/quarterly_nominal_wage_development.xlsx', header=3)
tidy_data = []

# Find the indices of the rows containing the actual data
data_row_indices = df[df.iloc[:,0] == 'Annual variation of nominal wages  (in %)'].index
data_row_indices
print(len(df.columns))

for idx in data_row_indices:
    year_row = df.iloc[idx - 4]
    quarter_row = df.iloc[idx - 3]
    value_row = df.iloc[idx]
    #Forward-fill the year values to handle merged cells
    year_row.ffill(inplace=True)
    #Iterate through the columns to extract each data point
    for col in range(1, len(df.columns)):
        year = year_row.iloc[col]
        quarter = quarter_row.iloc[col]
        value = value_row.iloc[col]
        print(year_row)
        tidy_data.append({'Year': int(year),'Quarter': quarter,'Nominal_Wage_Variation_Percent': value })

print(tidy_data)
#Create a DataFrame from the extracted list of dictionaries
wages = pd.DataFrame(tidy_data)
#errors into NaN
wages['Wage'] = pd.to_numeric(wages['Nominal_Wage_Variation_Percent'], errors='coerce')
#starting month of the quarter
quarter_to_month_map = {'I': 1, 'II': 4, 'III': 7, 'IV': 10}
wages['Month'] = wages['Quarter'].map(quarter_to_month_map)
wages['Day'] = 1 
#single datetime column
wages['Date'] = pd.to_datetime(wages[['Year', 'Month', 'Day']])
#monthly forward fill
wages = wages[['Date', 'Wage_change']].set_index('Date')
wages = wages.resample('MS').ffill()
wages

QRF_data= pd.merge(QRF_data, wages, left_index=True, right_index=True, how='left')



#Industrial Production: use total turnover as a proxy for industrial production, as production data is not available for the wished time frame 
url = 'https://www.pxweb.bfs.admin.ch/api/v1/en/px-x-0603010000_102/px-x-0603010000_102.px'

json_query = {
  "query": [
    {"code": "Bereinigung", "selection": {"filter": "item", "values": ["sa"]}},
    {"code": "Indizes / Veränderungsraten", "selection": {"filter": "item", "values": ["ind"]}},
    {"code": "Variable", "selection": {"filter": "item", "values": ["utot"]}},
    {"code": "Branche", "selection": {"filter": "item", "values": ["C"]}}
  ],
  "response": {"format": "json-stat"}
}
response = requests.post(url, json=json_query)
turnover = pyjstat.from_json_stat(response.json())[0]
turnover = turnover[['Quarter','value']]

#want to convert to monthly data using forward fill-> set first of month for easy conversion
turnover['Date'] = turnover['Quarter'].str.replace('Q1', '-01-01').str.replace('Q2', '-04-01').str.replace('Q3', '-07-01').str.replace('Q4', '-10-01')
turnover['Date'] = pd.to_datetime(turnover['Date'])
turnover = turnover.set_index('Date')
turnover = turnover.resample('MS').ffill()
turnover
#need PPI as Turnover is a nominal value makes it harder for your model to disentangle the real economic signal -> for production representation should deflate the turnover data.
PPI = pd.read_excel(file_path + '/PPI.xlsx', sheet_name= 'INDEX_m',header=6, skiprows=range(7, 80))
PPI=PPI[0:1195]
PPI= PPI[['Datum', 'Dez 2020 = 100']]
PPI['Datum'] = pd.to_datetime(PPI['Datum'])
PPI.columns=['Date','PPI']
PPI=PPI.set_index('Date')
turnover = turnover.merge(PPI, left_index=True, right_index=True, how='left')
turnover
# Deflate the turnover
turnover['real_turnover'] = (turnover['value'] / turnover['PPI']) * 100 
turnover=turnover[['PPI', 'real_turnover']]
QRF_data= pd.merge(QRF_data, turnover, left_index=True, right_index=True)
QRF_data

#Retail turnover
url2 = 'https://www.pxweb.bfs.admin.ch/api/v1/en/px-x-0603020000_101/px-x-0603020000_101.px'
json_query2 = {
  "query": [
    {"code": "Bereinigung", "selection": {"filter": "item", "values": ["sa"]}},
    {"code": "Indizes / Veränderungen", "selection": { "filter": "item","values": ["ind"]}},
    {"code": "Nominal / Real","selection": {"filter": "item","values": ["r"]}},
    {"code": "Branche / Warengruppe","selection": {"filter": "item","values": ["47"]}}],
  "response": {"format": "json-stat"}
}
response2 = requests.post(url2, json=json_query2)
retail = pyjstat.from_json_stat(response2.json())[0]
retail=retail[['Month', 'value']]
retail['Month'] = retail['Month'].astype(str)
retail['Date'] = retail['Month'].str.replace('M', '-')
retail['Date'] = pd.to_datetime(retail['Date'])
retail=retail[['value', 'Date']]
retail.columns=['retail_turnover', 'Date']
retail= retail.set_index('Date')
retail
QRF_data= pd.merge(QRF_data, retail, left_index=True, right_index=True, how='left')
QRF_data


#EURO exchange rate
Exchange = pd.read_excel(file_path+'/SNB_Exchange_rates.xlsx', header= 15)
Exchange=Exchange.iloc[:, [0, 1]]
Exchange.columns=['Date', 'Exchange_Rate_CHF']
Exchange= Exchange.set_index('Date')
QRF_data= pd.merge(QRF_data, Exchange, left_index=True, right_index=True, how='left')
QRF_data

#mortgages (proxy for Consumer credit lending)
mortgages= pd.read_csv(file_path+ '/Variable_mortgages.csv', header=2, sep=';')
mortgages= mortgages[['Date','Value']]
mortgages['Date'] = pd.to_datetime(mortgages['Date'])
mortgages.columns=['Date', 'variable_mortgages']
mortgages= mortgages.set_index('Date')
mortgages
QRF_data= pd.merge(QRF_data, mortgages, left_index=True, right_index=True, how='left')

#Volume of credit being extended domestically
Vol_loans= pd.read_csv(file_path +'/volume_loans.csv',header=2, sep=';')
Vol_loans= Vol_loans[['Date', 'Value']]
Vol_loans['Date']= pd.to_datetime(Vol_loans['Date'])
Vol_loans.columns=['Date', 'Vol_loans']
Vol_loans = Vol_loans.set_index('Date')
Vol_loans
QRF_data= pd.merge(QRF_data, Vol_loans, left_index=True, right_index=True, how='left')


#Money Supply changes (monthly)
Money_sup= pd.read_csv(file_path+'/Monetary_aggregate_change.csv', header=2, sep=';')
Money_sup['Date']= pd.to_datetime(Money_sup['Date'])
#long to wide
Money_sup = Money_sup.pivot(index='Date', columns='D1', values='Value')
Money_sup.columns= ['M1_change', 'M2_change', 'M3_change']
Money_sup 
QRF_data= pd.merge(QRF_data, Money_sup, left_index=True, right_index=True, how='left')

#Import Price Index by Origin (NAICS): Manufacturing for European Union
Manuf= pd.read_csv(file_path + '/Manufacturing_for_EU.csv')
Manuf['observation_date']= pd.to_datetime(Manuf['observation_date'])
Manuf.columns= ['Date', 'Manufacturing_EU']
Manuf= Manuf.set_index('Date')
QRF_data= pd.merge(QRF_data, Manuf, left_index=True, right_index=True, how='left')

#Business Confidence in EUro area
B_Conf=pd.read_csv(file_path+ '/Business_Confidence_EU.csv')
B_Conf['observation_date']= pd.to_datetime(B_Conf['observation_date'])
B_Conf.columns= ['Date', 'Business_Confidence_EU']
B_Conf= B_Conf.set_index('Date')
QRF_data= pd.merge(QRF_data, B_Conf, left_index=True, right_index=True, how='left')


#threshold variables
data = pxpy.PXDataset(file_path+ '/Industrial_Production.px')




#default facories fürs modeln
#hyperparameter with optuna

