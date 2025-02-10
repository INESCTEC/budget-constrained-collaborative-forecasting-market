'''
Forecasts released at day D 00:00 
Goal: forecast 24h-ahead (i.e. day D 1h00, 2h00, ..., 23h, 24h)

For such a scenario we have:
TARGET    | LAG           | LAG         | LAG         | WEATHER
D+1 01:00 | DAY D+1 00:00 | DAY D 23:00 | DAY D 22:00 | U,V
D+1 02:00 | DAY D+1 00:00 | DAY D 23:00 | DAY D 22:00 | U,V
D+1 03:00 | DAY D+1 00:00 | DAY D 23:00 | DAY D 22:00 | U,V
D+1 04:00 | DAY D+1 00:00 | DAY D 23:00 | DAY D 22:00 | U,V
...
'''

import pandas as pd
from datetime import datetime
from tqdm import tqdm # para ver barra de progresso em ciclos

# %% LOAD DATA
dateparse = lambda x: datetime.strptime(x, '%Y%m%d %H:%M')

exog = pd.read_csv('../data/Task15_Exogenous.csv', parse_dates= ['TIMESTAMP'], index_col= ['TIMESTAMP'])
exog.columns = [x.replace('Z', 'ZONE') for x in exog.columns.to_list()]
endog = pd.read_csv('../data/Task15_Endogenous.csv', parse_dates= ['TIMESTAMP'], index_col= ['TIMESTAMP'])

# %% PRE-PROCESS NA'S
endog.isna().sum()
exog.isna().sum() # No NAs found!
# TWO OPTIONS FOR NA REPLACEMENT:
# 1. NA=> MEAN VALUE OF THE COLUMN
# endog_pp = endog.fillna(endog.mean())
# 2. NA=> PREVIOUS VALUE != NA
endog = endog.ffill()
endog.isna().sum() # everything ok!
print('Data between', str(min(endog.index)), 'and', str(max(endog.index)))

# %% CONSTRUCT DATASETS
# create folder to save datasets:
folder_name = '../data/gefcom2014-processed'
import os
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

hours_previous_day = [23, 22, 21, 20, 19] # hours from previous day
max_horizon_with_lags = 6 # horizons using power lags

for forecast_horizon in range(1,max_horizon_with_lags+2):
    print('Train forecast horizon:'+str(forecast_horizon) + ' of ' + str(max_horizon_with_lags+1))
    if forecast_horizon <= max_horizon_with_lags:
        df_only_lags = pd.DataFrame([])
        for obs_ in tqdm(range(endog.shape[0])):
            
            Y = endog[endog.index==endog.index[obs_]]
            Y.columns = [i + '_Y' for i in Y.columns.to_list()]
            
            # midnight lag
            day_00 = pd.DataFrame(endog[endog.index==(endog.index[obs_] - pd.DateOffset(hours=forecast_horizon))])
            
            if len(day_00) >0:
                day_00.columns = [i + '_day_00' for i in day_00.columns.to_list()]
                day_00.index = Y.index
                new_obs = pd.concat([Y, day_00], axis=1)
            else:
                new_obs = Y
            
            # day D 23:00, 22:00, ...            
            for hour_ in range(1, len(hours_previous_day[:(max_horizon_with_lags-forecast_horizon-1)])+1):
                previous_day_hour_ = endog[endog.index==endog.index[obs_] - pd.DateOffset(hours=forecast_horizon+hour_)]
                if len(previous_day_hour_)>0:
                    previous_day_hour_.columns = [i + '_previous_day_'+str(hours_previous_day[hour_-1]) for i in previous_day_hour_.columns.to_list()]
                    previous_day_hour_.index = Y.index
                    new_obs = pd.concat([new_obs, previous_day_hour_], axis=1)
        
            df_only_lags = pd.concat([df_only_lags, new_obs])
    
        df_only_lags = pd.concat([df_only_lags, exog], axis=1)
        df_only_lags = df_only_lags.dropna()        
    else:
        Y = endog.copy()
        Y.columns = [i + '_Y' for i in Y.columns.to_list()]
        df_only_lags = pd.concat([Y, exog], axis=1)
        df_only_lags = df_only_lags.dropna()
    
    test_start = pd.to_datetime('2013-01-01 01:00:00') 
    
    # each month in 2013 is a test set
    for test_month in range(1,12):
        print('Test month:' + str(test_month))
        for zone in tqdm(range(1,11)):
            Y = df_only_lags[df_only_lags.columns[df_only_lags.columns.str.contains("ZONE"+str(zone)+"_Y", case=False)]]
            X = df_only_lags[df_only_lags.columns[~df_only_lags.columns.str.contains("_Y")]]
            Xown = df_only_lags[df_only_lags.columns[df_only_lags.columns.str.contains("ZONE"+str(zone)+"_") &
                                                      ~df_only_lags.columns.str.contains("_Y")]]
            train_start = test_start - pd.DateOffset(months=12)
            Ytr = Y[(Y.index>=train_start) & (Y.index<test_start)]
            Xtr = X[(X.index>=train_start) & (X.index<test_start)]
            Xown_tr = Xown[(Xown.index>=train_start) & (Xown.index<test_start)]
            
            common_name = f'_set{test_month}_zone{zone}_hour{forecast_horizon}.csv'
            Xtr.to_csv(f'{folder_name}/Xtrain{common_name}', index=True)
            Xown_tr.to_csv(f'{folder_name}/Xowntrain{common_name}', index=True)
            Ytr.to_csv(f'{folder_name}/Ytrain{common_name}', index=True)
            
            test_end = test_start + pd.DateOffset(months=1)
            
            if forecast_horizon > max_horizon_with_lags:
                Yts = Y[(Y.index>=test_start) & (Y.index<test_end) & ((Y.index.hour>=max_horizon_with_lags) | (Y.index.hour==0))]
                Xts = X[(X.index>=test_start) & (X.index<test_end) & ((X.index.hour>=max_horizon_with_lags) | (X.index.hour==0))]
                Xown_ts = Xown[(Xown.index>=test_start) & (Xown.index<test_end) & ((Xown.index.hour>=max_horizon_with_lags) | (Xown.index.hour==0))]
            else:
                Yts = Y[(Y.index>=test_start) & (Y.index<test_end) & (Y.index.hour==forecast_horizon)]
                Xts = X[(X.index>=test_start) & (X.index<test_end) & (X.index.hour==forecast_horizon)]
                Xown_ts = Xown[(Xown.index>=test_start) & (Xown.index<test_end) & (Xown.index.hour==forecast_horizon)]
            Xts.to_csv(f'{folder_name}/Xtest{common_name}', index=True)
            Xown_ts.to_csv(f'{folder_name}/Xowntest{common_name}', index=True)
            Yts.to_csv(f'{folder_name}/Ytest{common_name}', index=True)
        
            test_end = test_start + pd.DateOffset(months=1)            
            
        test_start = test_start + pd.DateOffset(months=1)
        
