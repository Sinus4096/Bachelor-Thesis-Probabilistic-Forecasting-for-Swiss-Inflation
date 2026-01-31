import pandas as pd
import glob
import os

# Define the paths to your results folders
folders = ['Results/Data_experiments', 'Results/Data_experiments2']
all_results = []

for folder in folders:
    # Grab all CSV files in the folder
    file_paths = glob.glob(os.path.join(folder, "*.csv"))
    
    for path in file_paths:
        file_name = os.path.basename(path)
        df = pd.read_csv(path)
        
        # Calculate mean scores as done in the paper [cite: 70, 260]
        # Using 'Empirical_CRPS' and 'Parametric_CRPS' to compare methods [cite: 177, 573]
        metrics = {
            'Folder': folder.split('/')[-1],
            'File': file_name,
            'Mean_Empirical_CRPS': df['Empirical_CRPS'].mean(),
            'Mean_Parametric_CRPS': df['Parametric_CRPS'].mean(),
            'Mean_Squared_Error': df['Squared_Error'].mean(),
            'RMSE': (df['Squared_Error'].mean())**0.5 # Paper uses RMSE for point forecasts [cite: 380]
        }
        all_results.append(metrics)

# Create a summary table
summary_df = pd.DataFrame(all_results)

# Sort for better scannability (Headline vs Core and Horizons)
summary_df = summary_df
print(summary_df.to_string(index=False))

