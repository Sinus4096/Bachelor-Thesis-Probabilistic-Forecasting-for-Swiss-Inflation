import requests
import pandas as pd
from pyjstat import pyjstat
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import os
warnings.filterwarnings('ignore')


script_dir = Path(__file__).resolve().parent
    # Go up to Code folder, then up to project root
BASE_PATH = script_dir.parent.parent / 'Code'/ 'Data' / 'Raw_Data'

def load_cpi_data(file_path):
    """Load and process CPI inflation data"""
    print("Loading CPI data...")
    cpi_data_wide = pd.read_excel(file_path / 'Inflation.xlsx', sheet_name='VAR_m-12', header=3)
    target_rows = ['Total', '    Kerninflation 1']
    filtered_data = cpi_data_wide[cpi_data_wide['Position_D'].isin(target_rows)].copy()
    filtered_data = filtered_data.drop_duplicates()
    
    id_columns = ['Position_D']
    data_long = filtered_data.melt(
        id_vars=id_columns, 
        var_name='Date', 
        value_name='Inflation'
    )
    data_long = data_long.iloc[26:]
    data_long['Date'] = pd.to_datetime(data_long['Date'])
    
    final_cpi_df = data_long.pivot(
        index='Date',
        columns='Position_D',
        values='Inflation'
    ).reset_index()
    final_cpi_df.columns = ['Date', 'Core_CPI', 'Headline_CPI']
    return final_cpi_df.set_index('Date')

def load_kof_barometer(file_path):
    """Load KOF Economic Barometer"""
    print("Loading KOF Barometer...")
    KOF_barometer = pd.read_excel(file_path / 'KOF_economic_barometer.xlsx')
    KOF_barometer['Date'] = pd.to_datetime(KOF_barometer['date'])
    KOF_barometer = KOF_barometer.drop('date', axis=1)
    return KOF_barometer.set_index('Date')

def load_unemployment(file_path):
    """Load and process unemployment rate"""
    print("Loading Unemployment data...")
    u_rate = pd.read_excel(file_path / 'Arbeitslosenquote.xlsx', sheet_name=None, header=4)
    data_list = []
    
    for year, df in u_rate.items():
        for i in range(len(df)):
            cell_value = df.iloc[i, 0]
            if pd.notna(cell_value) and str(cell_value).strip().lower() == 'total':
                total_row = df.iloc[i]
                for col in df.columns[2:]:
                    value = total_row[col]
                    if pd.notna(value):
                        date_str = f"{col}"
                        data_list.append({'date': date_str, 'total': value})
                break
    
    Unemployed = pd.DataFrame(data_list)
    Unemployed['date'] = pd.to_datetime(Unemployed['date'])
    Unemployed = Unemployed.sort_values('date').reset_index(drop=True)
    Unemployed.set_index('date', inplace=True)
    Unemployed.columns = ['unemployment_rate']
    return Unemployed

def load_oil_prices(file_path):
    """Load FRED oil prices"""
    print("Loading Oil prices...")
    oil_prices = pd.read_excel(file_path / 'FRED_Crude_Oil_Prices.xlsx', sheet_name='Monthly')
    oil_prices['Date'] = pd.to_datetime(oil_prices['observation_date'])
    oil_prices = oil_prices.drop('observation_date', axis=1)
    oil_prices.set_index('Date', inplace=True)
    oil_prices.columns = ['oilprices']
    return oil_prices

def load_gdp_ch(file_path):
    """Load Swiss GDP"""
    print("Loading Swiss GDP...")
    gdp = pd.read_excel(file_path / 'gdp.xlsx', sheet_name='real_q', header=10)
    gdp = gdp.iloc[:, [0, 1, 3]]
    gdp.columns = ['Year', 'Quarter', 'real gdp growth']
    gdp['Date'] = pd.to_datetime(gdp['Year'].astype(str) + 'Q' + gdp['Quarter'].astype(str))
    gdp = gdp.drop(['Quarter', 'Year'], axis=1)
    gdp.set_index('Date', inplace=True)
    gdp_CH = gdp.resample('MS').ffill()
    return gdp_CH

def load_gdp_eu(file_path):
    """Load EU GDP"""
    print("Loading EU GDP...")
    gdp2 = pd.read_excel(file_path / 'gdp_EU.xlsx', sheet_name='Quarterly')
    gdp2['gdp_EU_growth'] = gdp2['CLVMNACSCAB1GQEU272020'].pct_change()
    gdp2 = gdp2.drop('CLVMNACSCAB1GQEU272020', axis=1)
    gdp2.columns = ['Date', 'gdp_EU_growth']
    gdp2.set_index('Date', inplace=True)
    gdp_EU = gdp2.resample('MS').ffill()
    return gdp_EU

def load_inflation_expectations(file_path):
    """Load KOF inflation expectations"""
    print("Loading Inflation expectations...")
    infl_exp = pd.read_excel(file_path / 'KOF_consensus_forecast.xlsx')
    infl_exp = infl_exp[['date', 'ch.kof.consensus.q_qn_prices_cy.mean', 'ch.kof.consensus.q_qn_prices_ny.mean']]
    infl_exp.columns = ['Date', 'infl_e_current_year', 'infl_e_next_year']
    infl_exp['Date'] = pd.to_datetime(infl_exp['Date'])
    infl_exp.set_index('Date', inplace=True)
    monthly_range = pd.date_range(start='2001-01-01', end='2025-06-01', freq='MS')
    infl_exp = infl_exp.reindex(monthly_range).fillna(method='bfill')
    return infl_exp

def load_interest_rates(file_path):
    """Load and process interest rates and financial spreads"""
    print("Loading Interest rates...")
    st_libor = pd.read_excel(file_path / 'SNB_3-month_CHF_Libor_shortterm.xlsx', header=15)
    st_libor.columns = ['Date', 'Saron_Rate', 'Call_money_rate', 'Governmental_claims', '3m_CHF_Libor']
    st_libor['Date'] = pd.to_datetime(st_libor['Date'])
    st_libor.set_index('Date', inplace=True)
    
    lr_interest = pd.read_excel(file_path / 'SNB_Spot_interest_rates_longterm.xlsx', header=15)
    lr_interest.columns = ['Date', 'EU_2int', 'EU_10int', 'CH_2int', 'CH_10int']
    lr_interest['Date'] = pd.to_datetime(lr_interest['Date'])
    lr_interest.set_index('Date', inplace=True)
    lr_interest = lr_interest.resample('MS').mean()
    
    interest = pd.concat([st_libor, lr_interest], axis=1, join='outer')
    interest['fin_spread'] = interest['CH_10int'] - interest['Saron_Rate']
    interest = interest[['Saron_Rate', 'CH_2int', 'fin_spread']]
    return interest

def load_eu_interest(file_path):
    """Load EU interest rates"""
    print("Loading EU Interest rates...")
    EU_interest = pd.read_csv(file_path / 'Euro_Area_Interest.csv')
    EU_interest.columns = ['Date', 'EU_int']
    EU_interest['Date'] = pd.to_datetime(EU_interest['Date'])
    
    EU_short_term_int = pd.read_csv(file_path / 'Euribor_Euro_area.csv')
    EU_short_term_int.columns = ['Date', 'Time', 'short_term_int_EU']
    EU_short_term_int['Date'] = pd.to_datetime(EU_short_term_int['Date'])
    EU_short_term_int['Date'] = EU_short_term_int['Date'].dt.to_period('M').dt.to_timestamp()
    EU_short_term_int.set_index('Date', inplace=True)
    EU_interest.set_index('Date', inplace=True)
    
    EU_interest = pd.merge(EU_interest, EU_short_term_int, left_index=True, right_index=True, how='left')
    EU_interest['EU_fin_spread'] = EU_interest['EU_int'] - EU_interest['short_term_int_EU']
    return EU_interest[['EU_fin_spread']]

def load_wages(file_path):
    """Load wage data"""
    print("Loading Wage data...")
    df = pd.read_excel(file_path / 'quarterly_nominal_wage_development.xlsx', header=3)
    tidy_data = []
    
    data_row_indices = df[df.iloc[:, 0] == 'Annual variation of nominal wages  (in %)'].index
    
    for idx in data_row_indices:
        year_row = df.iloc[idx - 4]
        quarter_row = df.iloc[idx - 3]
        value_row = df.iloc[idx]
        year_row.ffill(inplace=True)
        
        for col in range(1, len(df.columns)):
            year = year_row.iloc[col]
            quarter = quarter_row.iloc[col]
            value = value_row.iloc[col]
            tidy_data.append({'Year': int(year), 'Quarter': quarter, 'Nominal_Wage_Variation_Percent': value})
    
    wages = pd.DataFrame(tidy_data)
    wages['Wage_change'] = pd.to_numeric(wages['Nominal_Wage_Variation_Percent'], errors='coerce')
    quarter_to_month_map = {'I': 1, 'II': 4, 'III': 7, 'IV': 10}
    wages['Month'] = wages['Quarter'].map(quarter_to_month_map)
    wages['Day'] = 1
    wages['Date'] = pd.to_datetime(wages[['Year', 'Month', 'Day']])
    wages = wages[['Date', 'Wage_change']].set_index('Date')
    wages = wages.resample('MS').ffill()
    return wages

def load_turnover_and_ppi(file_path):
    """Load industrial turnover and PPI via API"""
    print("Loading Industrial turnover (API)...")
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
    turnover = turnover[['Quarter', 'value']]
    
    turnover['Date'] = turnover['Quarter'].str.replace('Q1', '-01-01').str.replace('Q2', '-04-01').str.replace('Q3', '-07-01').str.replace('Q4', '-10-01')
    turnover['Date'] = pd.to_datetime(turnover['Date'])
    turnover = turnover.set_index('Date')
    turnover = turnover.resample('MS').ffill()
    
    PPI = pd.read_excel(file_path / 'PPI.xlsx', sheet_name='INDEX_m', header=6, skiprows=range(7, 80))
    PPI = PPI[0:1195]
    PPI = PPI[['Datum', 'Dez 2020 = 100']]
    PPI['Datum'] = pd.to_datetime(PPI['Datum'])
    PPI.columns = ['Date', 'PPI']
    PPI = PPI.set_index('Date')
    
    turnover = turnover.merge(PPI, left_index=True, right_index=True, how='left')
    turnover['real_turnover'] = (turnover['value'] / turnover['PPI']) * 100
    turnover = turnover[['PPI', 'real_turnover']]
    return turnover

def load_retail():
    """Load retail turnover via API"""
    print("Loading Retail turnover (API)...")
    url2 = 'https://www.pxweb.bfs.admin.ch/api/v1/en/px-x-0603020000_101/px-x-0603020000_101.px'
    json_query2 = {
        "query": [
            {"code": "Bereinigung", "selection": {"filter": "item", "values": ["sa"]}},
            {"code": "Indizes / Veränderungen", "selection": {"filter": "item", "values": ["ind"]}},
            {"code": "Nominal / Real", "selection": {"filter": "item", "values": ["r"]}},
            {"code": "Branche / Warengruppe", "selection": {"filter": "item", "values": ["47"]}}
        ],
        "response": {"format": "json-stat"}
    }
    response2 = requests.post(url2, json=json_query2)
    retail = pyjstat.from_json_stat(response2.json())[0]
    retail = retail[['Month', 'value']]
    retail['Month'] = retail['Month'].astype(str)
    retail['Date'] = retail['Month'].str.replace('M', '-')
    retail['Date'] = pd.to_datetime(retail['Date'])
    retail = retail[['value', 'Date']]
    retail.columns = ['retail_turnover', 'Date']
    return retail.set_index('Date')

def load_exchange_rate(file_path):
    """Load exchange rates"""
    print("Loading Exchange rates...")
    Exchange = pd.read_excel(file_path / 'SNB_Exchange_rates.xlsx', header=15)
    Exchange = Exchange.iloc[:, [0, 1]]
    Exchange.columns = ['Date', 'Exchange_Rate_CHF']
    return Exchange.set_index('Date')

def load_mortgages(file_path):
    """Load mortgage data"""
    print("Loading Mortgages...")
    mortgages = pd.read_csv(file_path / 'Variable_mortgages.csv', header=2, sep=';')
    mortgages = mortgages[['Date', 'Value']]
    mortgages['Date'] = pd.to_datetime(mortgages['Date'])
    mortgages.columns = ['Date', 'variable_mortgages']
    return mortgages.set_index('Date')

def load_volume_loans(file_path):
    """Load volume of loans"""
    print("Loading Volume of loans...")
    Vol_loans = pd.read_csv(file_path / 'volume_loans.csv', header=2, sep=';')
    Vol_loans = Vol_loans[['Date', 'Value']]
    Vol_loans['Date'] = pd.to_datetime(Vol_loans['Date'])
    Vol_loans.columns = ['Date', 'Vol_loans']
    return Vol_loans.set_index('Date')

def load_money_supply(file_path):
    """Load money supply data"""
    print("Loading Money supply...")
    Money_sup = pd.read_csv(file_path / 'Monetary_aggregate_change.csv', header=2, sep=';')
    Money_sup['Date'] = pd.to_datetime(Money_sup['Date'])
    Money_sup = Money_sup.pivot(index='Date', columns='D1', values='Value')
    Money_sup.columns = ['M1_change', 'M2_change', 'M3_change']
    return Money_sup

def load_manufacturing_eu(file_path):
    """Load EU manufacturing data"""
    print("Loading EU Manufacturing...")
    Manuf = pd.read_csv(file_path / 'Manufacturing_for_EU.csv')
    Manuf['observation_date'] = pd.to_datetime(Manuf['observation_date'])
    Manuf.columns = ['Date', 'Manufacturing_EU']
    return Manuf.set_index('Date')

def load_business_confidence_eu(file_path):
    """Load EU business confidence"""
    print("Loading EU Business Confidence...")
    B_Conf = pd.read_csv(file_path / 'Business_Confidence_EU.csv')
    B_Conf['observation_date'] = pd.to_datetime(B_Conf['observation_date'])
    B_Conf.columns = ['Date', 'Business_Confidence_EU']
    return B_Conf.set_index('Date')

def load_all_data_parallel(file_path):
    """Load all data sources in parallel for efficiency"""
    
    # Group functions by type: file-based vs API-based
    file_based_loaders = {
        'cpi': lambda: load_cpi_data(file_path),
        'kof_barometer': lambda: load_kof_barometer(file_path),
        'unemployment': lambda: load_unemployment(file_path),
        'oil_prices': lambda: load_oil_prices(file_path),
        'gdp_ch': lambda: load_gdp_ch(file_path),
        'gdp_eu': lambda: load_gdp_eu(file_path),
        'inflation_exp': lambda: load_inflation_expectations(file_path),
        'interest': lambda: load_interest_rates(file_path),
        'eu_interest': lambda: load_eu_interest(file_path),
        'wages': lambda: load_wages(file_path),
        'turnover_ppi': lambda: load_turnover_and_ppi(file_path),
        'exchange': lambda: load_exchange_rate(file_path),
        'mortgages': lambda: load_mortgages(file_path),
        'vol_loans': lambda: load_volume_loans(file_path),
        'money_supply': lambda: load_money_supply(file_path),
        'manufacturing_eu': lambda: load_manufacturing_eu(file_path),
        'business_conf_eu': lambda: load_business_confidence_eu(file_path),
    }
    
    # API-based loader (retail) - loaded separately
    api_loaders = {
        'retail': load_retail,
    }
    
    results = {}
    
    # Load file-based data in parallel
    print("\n=== Loading file-based data in parallel ===")
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_name = {executor.submit(loader): name for name, loader in file_based_loaders.items()}
        
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                results[name] = future.result()
                print(f"✓ {name} loaded successfully")
            except Exception as e:
                print(f"✗ Error loading {name}: {str(e)}")
                results[name] = None
    
    # Load API-based data
    print("\n=== Loading API-based data ===")
    for name, loader in api_loaders.items():
        try:
            results[name] = loader()
            print(f"✓ {name} loaded successfully")
        except Exception as e:
            print(f"✗ Error loading {name}: {str(e)}")
            results[name] = None
    
    return results

def merge_all_data(data_dict):
    """Merge all loaded datasets into single QRF_data DataFrame"""
    print("\n=== Merging all datasets ===")
    
    # Start with CPI data
    QRF_data = data_dict['cpi'].copy()
    
    # Define merge order (maintain original logic)
    merge_order = [
        'kof_barometer', 'unemployment', 'oil_prices', 'gdp_ch', 'gdp_eu',
        'inflation_exp', 'interest', 'eu_interest', 'wages', 'turnover_ppi',
        'retail', 'exchange', 'mortgages', 'vol_loans', 'money_supply',
        'manufacturing_eu', 'business_conf_eu'
    ]
    
    for key in merge_order:
        if data_dict[key] is not None:
            QRF_data = pd.merge(QRF_data, data_dict[key], left_index=True, right_index=True, how='left')
            print(f"✓ Merged {key}")
        else:
            print(f"✗ Skipped {key} (not loaded)")
    
    print(f"\nFinal dataset shape: {QRF_data.shape}")
    return QRF_data

# Main execution
if __name__ == "__main__":
    print("Starting data loading process...")
    print(f"Base path: {BASE_PATH}")
    
    # Load all data in parallel
    data_dict = load_all_data_parallel(BASE_PATH)
    
    # Merge into final dataset
    QRF_data = merge_all_data(data_dict)
    
    print("\n=== Data Loading Complete ===")
    print(QRF_data.info())
    print("\nFirst few rows:")
    print(QRF_data.head())
    
    # Optional: Save to file
    # QRF_data.to_csv('QRF_data.csv')
    # print("\nData saved to QRF_data.csv")

