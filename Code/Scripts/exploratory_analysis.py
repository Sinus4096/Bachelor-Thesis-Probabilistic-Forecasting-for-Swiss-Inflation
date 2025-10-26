from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Path is relative to the project root folder
path = 'Code/Data/Cleaned_Data/QRF_data.csv'

df = pd.read_csv(path, index_col='Date', parse_dates=True)

df.info()