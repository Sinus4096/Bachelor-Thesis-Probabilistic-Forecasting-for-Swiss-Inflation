import pandas as pd
import glob
import os

#define paths to  specific results folders (benchmarks and model variations)
folders=['Results/Data_experiments_benchmark','Results/Data_experiments_bvar', 'Results/Data_experiments_bvar2', 'Results/Data_experiments_qrf', 'Results/Data_experiments_qrf2']
all_results = []  #initialize list to store dictionary of metrics
#iterate through each experiment folder to collect results
for folder in folders:
    #grab all CSV files in the folder (each represents a target/horizon combination)
    file_paths=glob.glob(os.path.join(folder, "*.csv"))
    
    #iterate through individual result files
    for path in file_paths:
        #get filename for labeling source of  data
        file_name= os.path.basename(path)
        #load  results dataframe
        df=pd.read_csv(path)
        #calculate mean scores and PIT statistics to compare method performance
        metrics = {'Folder': folder.split('/')[-1], 'File': file_name, 'Mean_Empirical_CRPS': df['Empirical_CRPS'].mean(),
            'Mean_Parametric_CRPS': df['Parametric_CRPS'].mean(), 'PIT_Mean': df['PIT'].mean(), 'PIT_Std': df['PIT'].std()}
        #append metrics to  results list
        all_results.append(metrics)

#create a summary table for cross-model comparison
summary_df=pd.DataFrame(all_results)
#print the full table to console for immediate check of Headline vs Core results
print(summary_df.to_string(index=False))
#define output path for the summary table (to be used for paper tables)
save_name=f"Scripts/Plots_and_Tables/06a_mean_metrics_table.csv"
summary_df.to_csv(save_name) 
