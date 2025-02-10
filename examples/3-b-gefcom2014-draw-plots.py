import pandas as pd
import numpy as np
from plotnine import *

columns2select  = ['Set', 'Zone', 'Timestamp', 'LassoRegression',
       'SplineLasso', 'Kernel', 'GradientBoosting', 
       'LassoRegressionOwn', 'SplineLassoOwn', 'KernelOwn', 'GradientBoostingOwn']
df = pd.DataFrame()
# Loop through the specified zones and sets
for zone in [1,5,9]:
    for set_ in range(1, 12):
        # Read each CSV file into a DataFrame
        file_path = f'~/Desktop/ENERSHARE/cost-constrained/results/gefcom2014/part1-models-comparison/results_set{set_}_zone{zone}.csv'
        df_ = pd.read_csv(file_path, index_col=[0])[columns2select]
        
        # Append to the main DataFrame
        df = pd.concat([df, df_], ignore_index=True)

# Convert TIMESTAMP to datetime and extract the hour
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y-%m-%d %H:00:00')
df['Timestamp'] = df['Timestamp'].dt.hour
df['Zone'] = [f'Zone {x}' for x in df['Zone']]

# Square all columns (except TIMESTAMP, set, and zone), group by TIMESTAMP, set, and zone, and calculate the mean.
result = (
    df.set_index(['Timestamp', 'Zone'])   # Set grouping columns as the index temporarily
    .pow(2)                                      # Square all numeric columns
    .groupby(['Timestamp', 'Zone'])       # Group by TIMESTAMP, set, and zone
    .mean()                                      # Calculate the mean for each group
    .apply(np.sqrt)                              # Take the square root of each mean value
    .reset_index()                               # Reset index to make TIMESTAMP, set, and zone columns again
)

print(result)

# Reshape the DataFrame to long format
df_melt = pd.melt(result, id_vars=['Timestamp', 'Set','Zone'], 
                  var_name='variable', 
                  value_name='value')

df_melt['type_'] = np.where(df_melt['variable'].str.contains('Own'), 'Local', 'Collaborative')
df_melt['variable'] = df_melt['variable'].str.replace('Own', '', regex=False)
df_melt['variable'] = df_melt['variable'].str.replace('LassoRegression', 'LASSO', regex=False)
df_melt['variable'] = df_melt['variable'].str.replace('SplineLasso', 'SLASSO', regex=False)
df_melt['variable'] = df_melt['variable'].str.replace('GradientBoosting', 'GBR', regex=False)
df_melt['variable'] = df_melt['variable'].str.replace('Kernel', 'KR', regex=False)
print(df_melt)

df_melt['value'] = 100*df_melt['value']

# Create the plot
plot = (
    ggplot(df_melt) +
    geom_line(aes(x='Timestamp', y='value', color='variable', linetype='type_')) +
    facet_wrap('~Zone') +
    labs(x='Hour', y='RMSE (% of rated power)', color='', linetype='') +
    scale_color_manual(
        values={
            'LASSO': 'darkgreen',        # LASSO -> darkgreen
            'SLASSO': 'blue',            # SLASSO -> darkblue
            'KR': 'red',                 # KR -> red
            'GBRT': 'gray'               # GBRT -> gray
        }
    ) +
    theme_bw() +
    theme(legend_position='top',
          panel_background=element_rect(fill='white'),  # Sets the panel background to white
            panel_grid_major=element_line(color='gray', linetype='dashed', size=0.5),  # Optionally, set grid lines if desired
            #panel_grid_minor=element_line(color='gray', linetype='dashed', size=0.25),
            strip_background=element_rect(fill='white'),
            legend_key=element_blank(),
            text=element_text(color='black', size=11)) 
)

print(plot)

plot.save("../results/gefcom2014/gefcom2014_rmse.pdf", format="pdf", width=6, height=2.5)

result = (
    df.set_index(['Timestamp', 'Set', 'Zone'])   # Set grouping columns as the index temporarily
    .pow(2)                                      # Square all numeric columns
    .groupby(['Timestamp', 'Set', 'Zone'])       # Group by TIMESTAMP, set, and zone
    .mean()                                      # Calculate the mean for each group
    .apply(np.sqrt)                              # Take the square root of each mean value
    .reset_index()                               # Reset index to make TIMESTAMP, set, and zone columns again
)

result = result.loc[result['Zone']=='Zone 9']
result = result.loc[(result['Set']==1) | (result['Set']==2)  | (result['Set']==5)]
df_melt = pd.melt(result, id_vars=['Timestamp', 'Set','Zone'], 
                  var_name='variable', 
                  value_name='value')

df_melt['value'] = 100*df_melt['value']

df_melt['type_'] = np.where(df_melt['variable'].str.contains('Own'), 'Local', 'Collaborative')
df_melt['variable'] = df_melt['variable'].str.replace('Own', '', regex=False)
df_melt['variable'] = df_melt['variable'].str.replace('LassoRegression', 'LASSO', regex=False)
df_melt['variable'] = df_melt['variable'].str.replace('SplineLasso', 'SLASSO', regex=False)
df_melt['variable'] = df_melt['variable'].str.replace('GradientBoosting', 'GBR', regex=False)
df_melt['variable'] = df_melt['variable'].str.replace('Kernel', 'KR', regex=False)
print(df_melt)

df_melt['Set'] = [f'Test set {x}' for x in df_melt['Set']]
# Create the plot
plot = (
    ggplot(df_melt) +
    geom_line(aes(x='Timestamp', y='value', color='variable', linetype='type_')) +
    facet_wrap('~Set') +
    labs(x='Hour', y='RMSE (% of rated power)', color='', linetype='') +
    scale_color_manual(
        values={
            'LASSO': 'darkgreen',        # LASSO -> darkgreen
            'SLASSO': 'blue',            # SLASSO -> darkblue
            'KR': 'red',                 # KR -> red
            'GBRT': 'gray'               # GBRT -> gray
        }
    ) +
    theme_bw() +
    theme(legend_position='top',
          panel_background=element_rect(fill='white'),  # Sets the panel background to white
            panel_grid_major=element_line(color='gray', linetype='dashed', size=0.5),  # Optionally, set grid lines if desired
            #panel_grid_minor=element_line(color='gray', linetype='dashed', size=0.25),
            strip_background=element_rect(fill='white'),
            legend_key=element_blank(),
            text=element_text(color='black', size=11)) 
)

print(plot)
plot.save("../results/gefcom2014/rmse-plot-per-folder.pdf", format="pdf", width=6, height=2.5)

