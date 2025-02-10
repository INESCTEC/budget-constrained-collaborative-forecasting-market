import numpy as np
import time
import pandas as pd
from tqdm import tqdm # para ver barra de progresso em ciclos
from sklearn.linear_model import Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.kernel_ridge import KernelRidge

from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

from skopt.callbacks import DeltaXStopper
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import PredefinedSplit

cv_ = 11
n_initial_points = 30
n_calls = 60

def compute_models(Xtrain, Xtest, Ytrain, Ytest, cv_):
    names_input = Xtrain.columns.to_list()
    Ytrain = np.reshape(Ytrain, (len(Ytrain), 1))
    Ytest = Ytest
    # Configuration to run Bayesian optimization
    params_bopt = {'n_calls' : n_calls, 
                   'n_initial_points' : n_initial_points, 
                   'n_jobs':-1, 
                   'random_state':123, 
                   'callback': DeltaXStopper(delta=10**(-4)), 
                   'verbose': False}
    ## LASSO linear
    lasso = Pipeline(steps=[('scaler', StandardScaler()),
                           ('model', Lasso())])
    lasso_space = [Real(0.0001, 100, name='model__alpha')]
    @use_named_args(lasso_space)
    def lasso_objective(**params):
        lasso.set_params(**params)
        scores = (-1)*cross_val_score(lasso, Xtrain, Ytrain.ravel(), cv=cv_, scoring='neg_root_mean_squared_error', n_jobs=-1).mean()
        return scores
    t=time.time()
    lasso_res = gp_minimize(lasso_objective, lasso_space, **params_bopt)
    lasso_opt_params = lasso_res.x
    lasso.set_params(model__alpha=lasso_opt_params[0])
    lasso.fit(Xtrain, Ytrain.ravel())
    Ypred_lasso = lasso.predict(Xtest).reshape(len(Xtest),1)
    t_lasso = time.time()-t
    lasso_coefs = pd.DataFrame([lasso.named_steps['model'].coef_], columns = names_input)
    lasso_coefs['intercept'] = lasso.named_steps['model'].intercept_
    print(f'Lasso Optimization Done! Iters: {len(lasso_res.x_iters)}, Optimals: {lasso_res.x}')
    
    # LASSO spline 
    spline_lasso = Pipeline(steps=[('transformer', SplineTransformer()), 
                                   ('scaler', StandardScaler()),
                                   ('model', Lasso())])
    spline_lasso_space = [Integer(1, 5, name='transformer__degree'), 
                          Integer(3, 10, name='transformer__n_knots'), 
                          Real(0.0001, 100, name='model__alpha')]
    @use_named_args(spline_lasso_space)
    def spline_lasso_objective(**params):
        spline_lasso.set_params(**params)
        scores = (-1)*cross_val_score(spline_lasso, Xtrain, Ytrain.ravel(), cv=cv_, scoring='neg_root_mean_squared_error', n_jobs=-1).mean()
        return scores
    t=time.time()
    spline_lasso_res = gp_minimize(spline_lasso_objective, spline_lasso_space,**params_bopt)
    spline_lasso.set_params(**{spline_lasso_space[i].name: spline_lasso_res.x[i] for i in range(len(spline_lasso_res.x))})
    spline_lasso.fit(Xtrain, Ytrain.ravel())
    Ypred_spline_lasso = spline_lasso.predict(Xtest).reshape(len(Xtest),1)
    t_splasso = time.time()-t
    new_features_nr_ = spline_lasso_res.x[0] + spline_lasso_res.x[1] - 1
    splines_names = [f'{x}_s{s_}_' for x in names_input for s_ in range(1, new_features_nr_+1)]
    Slasso_coefs = pd.DataFrame([spline_lasso.named_steps['model'].coef_], columns = splines_names)
    Slasso_coefs['intercept'] = spline_lasso.named_steps['model'].intercept_
    print(f'Spline Lasso Optimization Done! Iters: {len(spline_lasso_res.x_iters)}, Optimals: {spline_lasso_res.x}')
    
    # Kernelized regression
    kernel = KernelRidge()
    kernel_space = [Real(1e-6, 100.0, 'log-uniform', name='alpha'),
                    Categorical(['rbf'], name='kernel'),
                    Real(1e-3, 100.0, 'log-uniform', name='gamma')]
    @use_named_args(kernel_space)
    def kernel_objective(**params):
        kernel.set_params(**params)
        scores = (-1)*cross_val_score(kernel, Xtrain, Ytrain.ravel(), cv=cv_, scoring='neg_root_mean_squared_error', n_jobs=1).mean()
        return scores
    t = time.time()
    kernel_res = gp_minimize(kernel_objective, kernel_space, **params_bopt)
    kernel.set_params(**{kernel_space[i].name: kernel_res.x[i] for i in range(len(kernel_res.x))})
    kernel.fit(Xtrain, Ytrain.ravel())
    Ypred_kernel = kernel.predict(Xtest).reshape(len(Xtest),1)
    t_kernel = time.time() - t
    print(f'Kernelized linear reg Optimization Done! Iters: {len(kernel_res.x_iters)}, Optimals: {kernel_res.x}')

    # GBR
    gbr = GradientBoostingRegressor()
    gbr_space = [Integer(3, 10, name='max_depth'),
                 Real(0.001,0.5, name='learning_rate'),
                 Integer(1,int(np.sqrt(Xtrain.shape[1])), name='max_features'),
                 Integer(10, 50, name='min_samples_split'),
                 Integer(10, 50, name='min_samples_leaf'),
                 Real(0.7,0.9, name='subsample'),
                 Integer(100,800, name='n_estimators')]
    @use_named_args(gbr_space)
    def gbr_objective(**params):
        gbr.set_params(**params)
        scores = (-1)*cross_val_score(gbr, Xtrain, Ytrain.ravel(), cv=cv_, scoring='neg_root_mean_squared_error', n_jobs=-1).mean()
        return scores
    t=time.time()
    gbr_res = gp_minimize(gbr_objective, gbr_space, **params_bopt)
    gbr_opt_params = gbr_res.x
    gbr.set_params(max_depth=gbr_opt_params[0],
                   learning_rate=gbr_opt_params[1],
                   max_features=gbr_opt_params[2],
                   min_samples_split=gbr_opt_params[3],
                   min_samples_leaf=gbr_opt_params[4],
                   subsample=gbr_opt_params[5],
                   n_estimators=gbr_opt_params[6])
    gbr.fit(Xtrain, Ytrain.ravel())
    Ypred_gbr = gbr.predict(Xtest).reshape(len(Xtest),1)    
    t_gbr = time.time() - t
    print(f'Gradient Boosting Optimization Done! Iters: {len(gbr_res.x_iters)}, Optimals: {gbr_res.x}')
    # Fit the models on the training data
    
    # Compute the root mean squared error for each model
    resid_lasso = (Ytest - Ypred_lasso)
    resid_splasso = (Ytest - Ypred_spline_lasso)
    resid_kernel = (Ytest -Ypred_kernel)
    resid_gbr = (Ytest - Ypred_gbr)
    
    residuals = np.concatenate((resid_lasso, resid_splasso, resid_kernel, resid_gbr), axis=1)
    df_residuals =  pd.DataFrame(residuals)
    df_residuals.columns = ['LASSO', 'SLASSO', 'KR', 'GBR']
    df_times = pd.DataFrame({'Lasso time': [t_lasso], 
                            'Spline Lasso time': [t_splasso], 
                            'Kernel time': [t_kernel],
                            'GBR time': [t_gbr]
                            })
    
    return df_residuals, df_times, lasso_coefs, Slasso_coefs

dateparse = '%Y-%m-%d %H:%M:%S'
max_hour = 6
path_ts='../data/gefcom2014-processed'
path_tr = '../data/gefcom2014-processed'

results_folder = '../results/gefcom2014/part1-models-comparison'
import os
if not os.path.exists(results_folder):
    os.makedirs(results_folder)
    
models_folder = '../results/gefcom2014/models-coefs'
if not os.path.exists(models_folder):
    os.makedirs(models_folder)

for zone in [9, 5, 1]: 
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
            # Compute models
            cv_= PredefinedSplit(np.array(Xtrain.index.month.to_list()))
            residuals_collab = compute_models(Xtrain, Xtest, Ytrain.to_numpy(), Ytest.to_numpy(), cv_)
            residuals_own = compute_models(Xtr_own, Xowntest, Ytrain.to_numpy(), Ytest.to_numpy(), cv_)
            
            collab = residuals_collab[0]
            own = residuals_own[0]
            own.columns = [f'{x}_own' for x in own.columns]
            results = pd.concat(((collab, own)), axis=1)
            results.index = Ytest.index
            
            results_per_zone_set = pd.concat((results_per_zone_set, results), axis=0)
            results_per_zone_set.to_csv(f'{results_folder}/zone{zone}_set{s}.csv')
            
            #results.to_csv(f'{results_folder}/zone{zone}_set{s}_hour{hour}.csv')
            
            lasso_coefs_own = residuals_collab[2]
            lasso_coefs_own.to_csv(f'{models_folder}/lasso_own_zone{zone}_set{s}_hour{hour}.csv')
            lasso_coefs = residuals_own[2]
            lasso_coefs.to_csv(f'{models_folder}/lasso_all_zone{zone}_set{s}_hour{hour}.csv')
            Slasso_coefs_own = residuals_collab[2]
            Slasso_coefs_own.to_csv(f'{models_folder}/Slasso_own_zone{zone}_set{s}_hour{hour}.csv')
            Slasso_coefs = residuals_own[2]
            Slasso_coefs.to_csv(f'{models_folder}/Slasso_all_zone{zone}_set{s}_hour{hour}.csv')
