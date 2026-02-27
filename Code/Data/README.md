## Data Sources and Processing

### Raw Data (`Code/Data/Raw_Data`)

The raw datasets used in this project originate from publicly available macroeconomic databases and official statistical institutions:

* **SNB (Swiss National Bank)**: interest rates (SARON, LIBOR, spot and long-term rates), exchange rates, monetary aggregates, mortgages, financial spreads.
* **FRED (Federal Reserve Economic Data)**: oil prices (Brent), EU manufacturing/output indicators, EU business confidence, GDP series and other international macroeconomic variables.
* **ECB (European Central Bank)**: financial spreads (e.g., EU yield spreads such as 10Y EU yield minus Euribor).
* **KOF Swiss Economic Institute**: KOF Economic Barometer and consensus inflation expectations (current and next year).
* **BFS / FSO (Swiss Federal Statistical Office)**: unemployment, producer price index (PPI), retail turnover, wage statistics, CPI (inflation).
* **SECO (State Secretariat for Economic Affairs)**: Swiss GDP and related macroeconomic indicators.
* **BES / other official statistical sources**: industrial turnover and additional real activity indicators where applicable.

All datasets are collected from official public sources and are used for academic research purposes only. Users should refer to the original providers for licensing terms and detailed metadata.

---

### Cleaned Data (`Code/Data/Cleaned_Data`)

* **`data_merged.csv`**
  All raw datasets after preprocessing, alignment, and merging into a single modeling dataset.

* **`data_stationary.csv`**
  Predictor dataset where variables are transformed to stationarity (e.g., log-differences, percentage changes) and inflation is de-annualized and shifted according to the modeling setup.

* **`data_stationary_bvar.csv`**
  Variant of the stationary dataset prepared for the BVAR specification. Lag structure is simplified (all lags are removed except the 1-month autoregressive lag).

* **`data_yoy.csv`**
  Year-over-year inflation dataset used as the final evaluation benchmark for forecast comparison.

---

### Notes

All processing scripts are available in the `Scripts` folder to ensure reproducibility.

