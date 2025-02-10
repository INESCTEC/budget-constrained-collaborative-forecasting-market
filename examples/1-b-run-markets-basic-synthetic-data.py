import numpy as np
import pandas as pd
import time

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.proposal import CostConstrainedGroupMarket

if not os.path.exists('../results'):
    os.makedirs('../results')

# MARKET CONFIGURATION
maxiter = 400
epsilon = 10**(-6)
cv_=10 
degree_bounds=[1,3] 
n_knots_bounds=[2,3]
alpha_bounds=[1e-2, 10]
n_calls = 30
n_initial_points = 15 
delta=1e-4 # tolerance for Bayesian optimization of splines order and regularization 
fs_filter='Pearson' # filter data according to Pearson correlation
fs_alpha = 0.05 # which p-value should we use to filter
gain_with_similar = False # use historical similar data to estimate gain? 


# Parameters related to feature importance (compare real model vs market model)
reps_ = 100
non_zero_index = [2, 6, 11, 20, 30, 36, 47, 50, 62, 89]
B_ = np.array([6, 5, 2, 10, 1, 4, 7, 3, 8, 9])
times_df = pd.DataFrame([], columns=['relation', 'time_bopt', 'time_fit', 'time_bgt', 'time_price'])

TableI = pd.DataFrame([])
for relation in ['linear', 'non-linear']:
    times_df_ = pd.DataFrame([np.zeros(5)], 
                             columns=['relation', 'time_bopt', 'time_fit', 'time_bgt', 'time_price'])
    times_df_['relation'] = relation
    Xown = pd.read_csv(f'../data/X_buyer-{relation}.csv')
    X_sellers = pd.read_csv(f'../data/X_sellers-{relation}.csv')
    X = pd.concat([Xown, X_sellers], axis=1)
    Y = pd.read_csv(f'../data/Y_buyer-{relation}.csv')
    
    real_coefs = pd.DataFrame(np.zeros((1,X.shape[1])), columns = X.columns)
    real_coefs.iloc[:, non_zero_index] = B_
    
    Xowntr = Xown.iloc[:6000,:] 
    Xtr = X.iloc[:6000,:] 
    Ytr = Y.iloc[:6000]
    Xownval = Xown.iloc[6000:8000,:] 
    Xval = X.iloc[6000:8000,:] 
    Yval = Y.iloc[6000:8000]
    Xownts = Xown.iloc[8000:,:] 
    Xts = X.iloc[8000:,:] 
    Yts = Y.iloc[8000:]
    
    # ---- RUN PROPOSAL
    # define costs per feature (original features)
    costs_per_group = np.array([10]*X.shape[1])
    costs_per_group[:10] = 0 # first 10 are assumed to be owned by data buyer!
    costs_per_group[36] = 11
    group_per_variable = np.arange(1,101,1) # each feature is a group
    budget = list(np.arange(1,100,1)) # possible set of final prices
    theta_0 = None # initial weights for the Bspline regression model
    
    start = time.time()
    # Initialize the market with the training set, costs, groups, budget, etc.
    proposal_market = CostConstrainedGroupMarket(Xowntr, Xtr, Ytr,
                                                 group_per_variable, 
                                                 costs_per_group, 
                                                 budget,
                                                 maxiter=maxiter, 
                                                 epsilon=epsilon,
                                                 theta_0=theta_0,
                                                 fs_filter=fs_filter,
                                                 fs_alpha=fs_alpha,
                                                 gain_with_similar=gain_with_similar)    
    # Tune parameters of the spline LASSO model (no contrains)
    # Local model
    # Collaborative model (all data available)
    proposal_market.tune_models(cv_ = 10, n_initial_points = 20, n_calls = 30)    
    # Use BO output to guide grid selection for the bid-constrained model
    grid_search = proposal_market.grid_search_[proposal_market.grid_search_[:, 3].argsort()[:1],:2]
    grid_search = np.unique(grid_search, axis=0)
    model_alpha = np.array([0.0001, 0.001, 0.01, 0.1, 1])
    repeated_A = np.repeat(grid_search, len(model_alpha), axis=0)
    tiled_C = np.tile(model_alpha, grid_search.shape[0])
    grid = np.column_stack((repeated_A, tiled_C))    
    proposal_market.tune_per_budget(group_per_variable, costs_per_group, grid, cv_)
    end = time.time()
    time_bopt = end - start 
    times_df_['time_bopt'] = time_bopt    
    
    # Fit cost-constrained spline LASSO model
    start = time.time()
    proposal_market.fit()
    end = time.time()
    time_fit = end - start 
    times_df_['time_fit'] = time_fit
    
    proposal_market.predict(Xval)
    
    # Construct the BGT with the fitted cost-constrained spline LASSO model
    start = time.time()
    proposal_market.BidGainTable(Xownval, Xval, Yval)
    end = time.time()
    time_bgt = end - start 
    times_df_['time_bgt'] = time_bgt
    
    # Evaluate different VF functions from data buyer
    for config in [1, 2]:
        print(f'config {config} relation {relation}')        
        # Initialize the market with the training set, costs, groups, budget, etc.        
        if config==1:
            def ValueFunction(gain): # how much agent is willing o pay for a a gain g
                return 50
        else:
            def ValueFunction(gain): # how much agent is willing o pay for a a gain g
                return 100
        # Find optimal price and corresponding model
        start = time.time()
        proposal_market.price(ValueFunction)
        end = time.time()
        time_price = end - start 
        times_df_['time_price'] = time_price        
        times_df = pd.concat((times_df, times_df_), axis=0)
        print(f'Final market price {proposal_market.final_price}')  
        
        ### PERMUTATION IMPORTANCE       
        # > Market model
        if proposal_market.gain_with_similar:
            thetas = proposal_market.final_model_coefs
        else:
            thetas = proposal_market.final_model_coefs
        
        original_features = [s.split('_s')[0] for s in thetas.index.get_level_values(0)]
        X_ = Xtr.copy()
        Y_ = Ytr.copy() 
        rmse_fi = pd.DataFrame(np.zeros((2, X.shape[1])))
        rmse_fi.columns = X.columns
        allocated_features = list(set([original_features[i] for i in np.where(thetas.values!=0)[0]]))
        allocated_features = [x for x in allocated_features if x!='intercept']
        pos_opt = np.where(proposal_market.budget==proposal_market.final_price)[0][0]
        Yhat_ = proposal_market.predict(Xtr).iloc[:, pos_opt]
        rmse_ = np.sqrt(np.mean((Y_.values[:,0] - Yhat_.values)**2))
        for i_f_, f_ in enumerate(allocated_features):
            Xshuf_ = X_.copy()
            if f_ not in Xown.columns:
                rmse_fi.loc[0, f_] = (-1)*rmse_
                for rep_ in range(reps_):
                    Xshuf_[f_] = np.random.permutation(X_[f_].values)
                    # market model
                    Yhat_shuf = proposal_market.predict(Xshuf_)[pos_opt]
                    rmse_shuf = np.sqrt(np.mean((Y_.values[:,0] - Yhat_shuf.values)**2)) 
                    rmse_fi.loc[0, f_] = rmse_fi.loc[0, f_] + (1/reps_)*rmse_shuf
        # > Real model
        Xtr.columns = Xtr.columns.get_level_values(0)
        Yhat_ = Xtr @ real_coefs.T
        if relation=='non-linear':
            Yhat_ = np.exp(0.05 * Yhat_)
        Yhat_.columns = Y_.columns
        rmse_ = np.sqrt(np.mean((Y_ - Yhat_)**2))
        for f_ in X.columns:
            if (f_ not in Xown.columns) and (real_coefs[f_].values[0]!=0):
                rmse_fi.loc[1, f_] = (-1)*rmse_
                Xshuf_ = X_.copy()
                for rep_ in range(reps_):
                    Xshuf_[f_] = np.random.permutation(X_[f_].values)
                    Yhat_shuf = Xshuf_ @ real_coefs.T
                    if relation=='non-linear':
                        Yhat_shuf = np.exp(0.05*Yhat_shuf)
                    Yhat_shuf.columns = Y_.columns
                    rmse_shuf = np.sqrt(np.mean((Y_ - Yhat_shuf)**2)) 
                    rmse_fi.loc[1, f_] = rmse_fi.loc[1, f_] + (1/reps_)*rmse_shuf
        fi_final = 100*rmse_fi.div(rmse_fi.sum(axis=1), axis=0).round(4)
        fi_final = fi_final.iloc[:,np.where(fi_final.sum(0)!=0)[0]] 
        # Forecasts for feature:
        if proposal_market.gain_with_similar:
            final_forecasts = proposal_market.final_forecasts
        else:
            final_forecasts = proposal_market.predict(Xts)[pos_opt]
            final_observed_gain = proposal_market.forecasting_gain(Xownts, Xts, Yts)
        
        case_description = f'../results/FS{fs_filter}-VF{config}-Similar{gain_with_similar}-{relation}'
        
        final_observed_gain.to_csv(f'{case_description}-bgt-observed.csv', index=True)
        proposal_market.bgt.to_csv(f'{case_description}-bgt-estimated.csv', index=True)
        TableI = pd.concat((TableI, fi_final.T), axis=1)

names_to_consider = ['real-1-linear','est-1-linear', 'est-2-linear',
                     'real-1-non-linear','est-1-non-linear', 'est-2-non-linear']
TableI = TableI[names_to_consider].sort_values(by=['real-1-linear'], ascending=False)
TableI.to_csv('../results/tableI.csv')
