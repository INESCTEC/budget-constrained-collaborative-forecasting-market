import numpy as np
import pandas as pd
from tqdm import tqdm # para ver barra de progresso em ciclos

from sklearn.model_selection import PredefinedSplit

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.proposal import CostConstrainedGroupMarket

import warnings
warnings.filterwarnings("ignore")

dateparse = '%Y-%m-%d %H:%M:%S'
max_hour = 6
path_ts='../data/gefcom2014-processed'
path_tr = '../data/gefcom2014-processed'

results_folder = '../results/gefcom2014/part2-data-markets'
import os
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

maxiter = 1000
epsilon = 10**(-3)
n_calls = 25
n_initial_points = 10
delta=1e-3
gain_with_similar = True
set_vfs = [300, 250, 200, 150, 100, 50, 20, 10]
fs_filter = None#'PartialPearson'
fs_alpha = 0.05

def ValueFunction1(gain): 
    return 100
def ValueFunction2(gain): 
    return 10
def ValueFunction3(gain): 
    return gain                
def ValueFunction4(gain): 
    return 40/(30 - gain) - 1.1               

for zone in range(1,11): 
    for s in range(1,12):  
        results_per_zone_set = pd.DataFrame()
        for hour in tqdm(range(1,max_hour+2)):        
            print(f' Running Zone {zone} - set {s} - hour {hour}...')
            # Training data
            params_read_file = {'parse_dates' : ['TIMESTAMP'], 'date_format': dateparse, 'index_col': ['TIMESTAMP']}
            Xtrain = pd.read_csv(path_tr+f'/Xtrain_set{s}_zone{zone}_hour{hour}.csv', **params_read_file)
            Ytrain = pd.read_csv(path_tr+f'/Ytrain_set{s}_zone{zone}_hour{hour}.csv', **params_read_file)
            Xtr_own = pd.read_csv(path_tr+f'/Xowntrain_set{s}_zone{zone}_hour{hour}.csv', **params_read_file)
            # Test data
            Xts = pd.read_csv(path_ts+f'/Xtest_set{s}_zone{zone}_hour{hour}.csv', **params_read_file)
            Yts = pd.read_csv(path_ts+f'/Ytest_set{s}_zone{zone}_hour{hour}.csv', **params_read_file)
            Xts_own = pd.read_csv(path_ts+f'/Xowntest_set{s}_zone{zone}_hour{hour}.csv', **params_read_file)
            # Adjust test data to this hour
            Xtest = Xts[Xtrain.columns]
            Xtest = Xtest[Xts.index.hour == hour]
            Ytest = Yts[Yts.index.hour == hour]
            Xowntest = Xts_own[Xtr_own.columns]
            Xowntest = Xowntest[Xts_own.index.hour == hour]   
            if hour==7:
                Xtest = Xts[Xtrain.columns]
                Xtest = Xtest[(Xts.index.hour >= hour) | (Xts.index.hour == 0)]
                Ytest = Yts[(Yts.index.hour >= hour) | (Yts.index.hour == 0)]
                Xowntest = Xts_own[Xtr_own.columns]
                Xowntest = Xowntest[(Xts_own.index.hour >= hour) | (Xts_own.index.hour == 0)]   
            # Compute fold for model tunning
            cv_= PredefinedSplit(np.array((Xtrain.index - pd.Timedelta(hours=1)).month.to_list()))
            # define costs per feature (original features)
            costs_per_group = np.array([1]*Xtrain.shape[1])
            zones  = Xtrain.columns.str.extract(r'(ZONE\d+)', expand=False).to_list()
            zone_indices = [i for i, zone_ in enumerate(zones) if zone_ == f'ZONE{zone}']
            costs_per_group[zone_indices] = 0 # owned by data buyer, no payment!
            # define groups per feature (here is the same as original features)
            group_per_variable = np.arange(1,Xtrain.shape[1]+1,1) # we assume each feature is a group
            budget = list(np.arange(1,costs_per_group.sum()+1,1)) # possible set of final prices
            beta_initial = None
            
            proposal_market = CostConstrainedGroupMarket(Xtr_own, Xtrain, Ytrain,
                                                        group_per_variable, 
                                                        costs_per_group, 
                                                        budget,
                                                        maxiter=maxiter, 
                                                        epsilon=epsilon,
                                                        beta_initial=beta_initial,
                                                        fs_filter=fs_filter,
                                                        fs_alpha=fs_alpha,
                                                        gain_with_similar=gain_with_similar)
            # Tune parameters of the spline LASSO model (no contrains)
            proposal_market.tune_models(cv_, n_initial_points, n_calls)
            # Use BO output to guide grid selection for the bid-constrained model
            grid_search = proposal_market.grid_search_[proposal_market.grid_search_[:, 3].argsort()[:1],:2]
            grid_search = np.unique(grid_search, axis=0)
            model_alpha = np.array([0.0000001, 0.0001, 0.001, 0.01, 0.1, 1])
            repeated_A = np.repeat(grid_search, len(model_alpha), axis=0)
            tiled_C = np.tile(model_alpha, grid_search.shape[0])
            grid = np.column_stack((repeated_A, tiled_C))
            proposal_market.tune_per_budget(group_per_variable, costs_per_group, grid, cv_)
            
            # Fit cost-constrained spline LASSO model
            proposal_market.fit()
            
            # Construct the BGT with the fitted cost-constrained spline LASSO model
            first_level = ['VF1', 'VF2', 'VF3', 'VF4']
            payments = pd.DataFrame(np.zeros((Xtest.shape[0], 4)), 
                                    columns=first_level,
                                    index = Xtest.index)
            gains = pd.DataFrame(np.zeros((Xtest.shape[0], 4)), 
                                 columns=first_level,
                                    index = Xtest.index)
            gains_estimated = pd.DataFrame(np.zeros((Xtest.shape[0], 4)), 
                                 columns=first_level,
                                    index = Xtest.index)
            second_level = [f'ZONE{i}' for i in range(1, 11)]  # Creates ZONE1 to ZONE10                    
            # Create a MultiIndex for the columns
            columns = pd.MultiIndex.from_product([first_level, second_level])                    
            # Create a DataFrame with this MultiIndex, filled with some sample data
            # For example, using NaN values or random numbers as initial values
            revenues = pd.DataFrame(np.zeros((Xtest.shape[0], 10*4)), 
                                    columns=columns,
                                    index = Xtest.index)
            for obs_ in Xtest.index:
                proposal_market.BidGainTable(Xowntest.loc[obs_].to_frame().T, 
                                             Xtest.loc[obs_].to_frame().T, 
                                             None,
                                             k=10)
                final_observed_gain = proposal_market.forecasting_gain(Xowntest.loc[obs_].to_frame().T, 
                                                                       Xtest.loc[obs_].to_frame().T, 
                                                                       Ytest.loc[obs_].to_frame().T)
                
                for vf_nr in range(1,4):
                    vf = globals()[f'ValueFunction{vf_nr}']
                    proposal_market.price(vf)
                    payments.loc[obs_, f'VF{vf_nr}'] = proposal_market.final_price[0]
                    if payments.loc[obs_, f'VF{vf_nr}']>0:
                        gains.loc[obs_, f'VF{vf_nr}'] = final_observed_gain['collab_rmse'].loc[final_observed_gain['bid']==proposal_market.final_price[0]].values[0]
                        gains_estimated.loc[obs_, f'VF{vf_nr}'] = proposal_market.bgt['gain'].loc[proposal_market.bgt['bid']==proposal_market.final_price[0]].values[0]
                        betas = proposal_market.estimated_coefs_market
                        pos_ = np.where(proposal_market.bgt['bid'] == proposal_market.final_price[0])
                        betas = betas[pos_[0][0]].copy()
                        betas = pd.DataFrame(betas.abs())
                        betas.columns = ['betas']
                        betas = betas.reset_index().iloc[:, 1:]
                        result_betas = betas.groupby('variable').agg(
                            sum_abs_betas=('betas', 'sum'),
                            price=('price_per_group', 'first')  # Use 'first' to get the unique price per variable
                        ).reset_index()
                        result_betas['ZONE'] = result_betas['variable'].str.extract(r'(ZONE\d+)', expand=False)
                        revenues_ = result_betas.groupby('ZONE')['price'].sum().reset_index()
                        revenues_.index= revenues_['ZONE']
                        revenues[f'VF{vf_nr}'].loc[obs_, :] = revenues_.loc[revenues[f'VF{vf_nr}'].columns.to_list(), 'price']
                
            payments.to_csv(f'{results_folder}/payments-Zone{zone}_Set{s}_hour{hour}.csv', index=True)
            revenues.to_csv(f'{results_folder}/revenues-paid-by-Zone{zone}_Set{s}_hour{hour}.csv', index=True)
            gains.to_csv(f'{results_folder}/gains-observed-Zone{zone}_Set{s}_hour{hour}.csv', index=True)
            gains_estimated.to_csv(f'{results_folder}/gains-estimated-Zone{zone}_Set{s}_hour{hour}.csv', index=True)
            
