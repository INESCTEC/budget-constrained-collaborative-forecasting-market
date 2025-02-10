import numpy as np
import pandas as pd
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.proposal import CostConstrainedGroupMarket
from src.custom_pipeline import CustomSplineLassoModelPipeline

import warnings
warnings.filterwarnings("ignore")

results_folder = '../results/'

# Covariance matrix construction
def generate_sparse_covariance_matrix(dim, sparsity=0.25, seed_=1):
    """
    Generate a sparse covariance matrix with a given fraction of zeros.
    
    Parameters:
    dim (int): Dimension of the covariance matrix (number of features).
    sparsity (float): Fraction of off-diagonal elements to set to zero (0 = no sparsity, 1 = fully sparse).
    
    Returns:
    cov_matrix (np.ndarray): Sparse covariance matrix of shape (dim, dim).
    """
    
    # Step 1: Create a random covariance matrix
    np.random.seed(seed_+1)                        
    A = np.random.randn(dim, dim)  # Generate a random matrix
    cov_matrix = A @ A.T  # Make it symmetric positive semi-definite
    # Step 2: Calculate the exact number of off-diagonal elements to zero out
    num_off_diag_elements = dim * (dim - 1) // 2  # Total off-diagonal elements in upper triangle
    num_zeros = int(sparsity * num_off_diag_elements)  # Number of elements to set to zero
    # Step 3: Create a list of all possible off-diagonal indices in the upper triangle
    upper_tri_indices = np.triu_indices(dim, k=1)  # Indices of the upper triangle (excluding diagonal)
    # Step 4: Randomly select positions to set to zero
    np.random.seed(seed_+2)                        
    zero_indices = np.random.choice(np.arange(num_off_diag_elements), size=num_zeros, replace=False)
    # Step 5: Create a mask where the selected off-diagonal elements are set to zero
    mask = np.ones((dim, dim), dtype=bool)  # Start with a mask of all ones
    mask[upper_tri_indices[0][zero_indices], upper_tri_indices[1][zero_indices]] = 0  # Zero out selected positions
    # Reflect the mask to the lower triangle to maintain symmetry
    mask = np.triu(mask) + np.triu(mask, 1).T  # Ensure symmetry
    # Ensure the diagonal stays intact (diagonal elements must be non-zero)
    np.fill_diagonal(mask, 1)
    # Apply the mask to make the matrix sparse
    sparse_cov_matrix = cov_matrix * mask
    return sparse_cov_matrix

if __name__ == '__main__':
    sparsity_ = [0.25, 0.5]#[0, 0.25, 0.5, 0.75]
    redundancy_ =  [0, 0.25]
    sellers_ = [200, 300, 400, 500] #100
    repetitions = 5
    results_proposal = [] # save all results
    maxiter = 200
    epsilon = 10**(-3)
    cv_=10 
    degree_bounds=[1,3] 
    n_knots_bounds=[2,3]
    alpha_bounds=[1e-2, 10]
    n_calls = 20
    n_initial_points = 10
    delta=1e-3
    fs_alpha = 0.05
    gain_with_similar = False
    set_vfs = [300, 250, 200, 150, 100, 50, 20, 10]
    reps_ = 100 # for permutation importance on real model
    seed_ = 0
    times_df = pd.DataFrame([], columns=['relation','nsellers', 
                                         'sparsity', 'redundancy',
                                         'repetition', 'fsfilter',
                                         'time_bopt', 'time_fit', 
                                         'time_bgt', 'time_price',
                                         'bid', 'payment',
                                         'estimated gain', 'observed gain'])
    
    for nsellers in sellers_:
        for relation in ['linear']:#, 'non-linear']:
            for s_ in sparsity_:
                for r_ in redundancy_:
                    for repetition_ in range(repetitions):
                        print(f'Relation {relation}- sellers {nsellers} - sparsity {s_} - redundancy {r_}- rep {repetition_}')
                        times_df_ = pd.DataFrame([np.zeros(14)], columns=times_df.columns.to_list())
                        times_df_['relation'] = relation
                        times_df_['nsellers'] = nsellers
                        times_df_['sparsity'] = s_
                        times_df_['redundancy'] = r_
                        times_df_['repetition'] = repetition_
                        seed_ = seed_ +1
                        np.random.seed(seed_)
                        B_ = np.random.normal(0, 10, (nsellers, 1))
                        seed_ = seed_ +1
                        np.random.seed(seed_)
                        pos_ = np.random.choice(nsellers, int(s_*nsellers), replace=False)
                        B_[pos_] = 0                        
                        covariance_ = generate_sparse_covariance_matrix(nsellers, 1-r_, seed_)
                        seed_ = seed_ +1
                        np.random.seed(seed_)
                        X = np.random.multivariate_normal(mean=np.zeros(nsellers), cov=covariance_, size=(10000, ))
                        seed_ = seed_ +1
                        np.random.seed(seed_)
                        error_ = np.random.normal(0, 1, (10000, 1))
                        Y = X @ B_
                        if relation=='non-linear':
                            scale_exp = 5/(X @B_).max()
                            Y = np.exp(scale_exp * Y) 
                        Y = Y + error_
                        Y = pd.DataFrame(Y, columns = ['Y'])
                        X = pd.DataFrame(X)
                        X.columns = [f'X_{x}' for x in range(1, nsellers+1)]
                        Xown = X[[f'X_{x}' for x in range(1, 11)]]
                        Xowntr = Xown.iloc[:6000,:] 
                        Xtr = X.iloc[:6000,:] 
                        Ytr = Y.iloc[:6000]
                        Xownval = Xown.iloc[6000:8000,:] 
                        Xval = X.iloc[6000:8000,:] 
                        Yval = Y.iloc[6000:8000]
                        Xownts = Xown.iloc[8000:,:] 
                        Xts = X.iloc[8000:,:] 
                        Yts = Y.iloc[8000:]
                        real_coefs = pd.DataFrame([B_[:,0]], columns = X.columns)
                        # ---- RUN PROPOSAL
                        # define costs per feature (original features)
                        costs_per_group = np.array([1]*X.shape[1])
                        costs_per_group[:10] = 0 # first 10 are assumed to be owned by data buyer!
                        # define groups per feature (here is the same as original features)
                        group_per_variable = np.arange(1,nsellers+1,1) # we assume each feature is a group
                        budget = list(np.arange(1,costs_per_group.sum()+1,1)) # possible set of final prices
                        beta_initial = None
                        
                        for fs_filter in ['PartialPearson', None, 'Pearson']:
                            times_df_['fsfilter'] = fs_filter
                            start = time.time()
                            # Initialize the market with the training set, costs, groups, budget, etc.
                            proposal_market = CostConstrainedGroupMarket(Xowntr, Xtr, Ytr,
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
                            
                            # Construct the BGT with the fitted cost-constrained spline LASSO model
                            start = time.time()
                            proposal_market.BidGainTable(Xownval, Xval, Yval)
                            end = time.time()
                            time_bgt = end - start 
                            times_df_['time_bgt'] = time_bgt
                            
                            final_observed_gain = proposal_market.forecasting_gain(Xownts, Xts, Yts)
                                
                            case_description = f'{results_folder}FS{fs_filter}-Similar{gain_with_similar}-{relation}-{nsellers}-{s_}-{r_}-{repetition_}'
                            final_observed_gain.to_csv(f'{case_description}-bgt-observed.csv', index=True)
                            proposal_market.bgt.to_csv(f'{case_description}-bgt-estimated.csv', index=True)
                            
                            ### PERMUTATION IMPORTANCE
                            # Feature importance > Real model
                            rmse_fi = pd.DataFrame([np.zeros(X.shape[1])], columns = X.columns)
                            X_ = Xtr.copy()
                            Y_ = Ytr.copy() 
                            Xtr.columns = Xtr.columns.get_level_values(0) 
                            Yhat_ = Xtr @ real_coefs.T
                            if relation=='non-linear':
                                Yhat_ = np.exp(scale_exp * Yhat_)
                            Yhat_.columns = Y_.columns
                            rmse_ = np.sqrt(np.mean((Y_ - Yhat_)**2))
                            for f_ in X.columns:
                                if (f_ not in Xown.columns) and (real_coefs[f_].values[0]!=0):
                                    rmse_fi.loc[:, f_] = (-1)*rmse_
                                    Xshuf_ = X_.copy()
                                    for rep_ in range(reps_):
                                        Xshuf_[f_] = np.random.permutation(X_[f_].values)
                                        Yhat_shuf = Xshuf_ @ real_coefs.T
                                        if relation=='non-linear':
                                            Yhat_shuf = np.exp(scale_exp*Yhat_shuf)
                                        Yhat_shuf.columns = Y_.columns
                                        rmse_shuf = np.sqrt(np.mean((Y_ - Yhat_shuf)**2)) 
                                        rmse_fi[f_] = rmse_fi[f_] + (1/reps_)*rmse_shuf                            
                            
                            # Evaluate different VF functions from data buyer
                            rmse_fi_ = pd.DataFrame([np.zeros(X.shape[1])], columns = X.columns)
                            for bid_ in set_vfs: 
                                rmse_fi_ = pd.DataFrame([np.zeros(X.shape[1])], columns = X.columns)
                                times_df_['bid'] = bid_
                                # Initialize the market with the training set, costs, groups, budget, etc.        
                                def ValueFunction(gain): # how much agent is willing o pay for a a gain g
                                    return bid_    
                                # Find optimal price and corresponding model
                                start = time.time()
                                proposal_market.price(ValueFunction)
                                end = time.time()
                                time_price = end - start 
                                times_df_['time_price'] = time_price  
                                # > Market model
                                times_df_['payment'] = proposal_market.final_price
                                if proposal_market.final_price == None:
                                    pos_ = 0
                                else:
                                    pos_ = np.where(proposal_market.bgt['bid'] == proposal_market.final_price)[0][0]
                                times_df_['estimated gain'] = proposal_market.bgt.iloc[pos_, 1]
                                times_df_['observed gain'] = final_observed_gain.iloc[pos_, 1]
                                times_df = pd.concat((times_df, times_df_), axis=0)
                    
                    times_df.to_csv(f'{results_folder}times-advanced-synthetic.csv')
