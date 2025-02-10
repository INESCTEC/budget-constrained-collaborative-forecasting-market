import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt.callbacks import DeltaXStopper
from scipy.stats import pearsonr
from scipy.spatial.distance import cdist
import sympy as sp

from main.custom_pipeline import CustomSplineLassoModelPipeline


class FeatureSelectorByPValue(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.05):
        self.threshold = threshold
        self.selected_features_ = []
        
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.selected_features_ = []
        for i in range(X.shape[1]):
            p_value = pearsonr(X[:, i], y[:, 0]).pvalue
            if p_value < self.threshold:
                self.selected_features_.append(i)
        if len(self.selected_features_)==0:
            self.selected_features_ = [1]
        return self
    
    def transform(self, X):
        ret = X[:, self.selected_features_]
        return ret


def knapsack(w, p, cap):
    n = len(w)
    x = [False] * n
    F = np.zeros((cap+1, n))
    G = [0] * (cap + 1)

    for k in range(n):
        F[:, k] = G[:]
        H = [0] * w[k] + [num_ + p[k] for num_ in G[:(cap + 1 - w[k])]]
        G = [max(G[i], H[i]) for i in range(cap + 1)]

    fmax = G[cap]
    f = fmax
    j = cap

    for k in range(n - 1, -1, -1):
        if F[j][k] < f:
            x[k] = True
            j -= w[k]
            f = F[j][k]
    
    inds = [i for i in range(len(x)) if x[i]]
    wght = sum(w[i] for i in inds)
    prof = sum(p[i] for i in inds)

    return {'capacity': wght, 'profit': prof, 'indices': inds}

def subset_sum(numbers, target): 
    # function to define budgets that can be allocated
    n = len(numbers)
    dp = [[False] * (target + 1) for _ in range(n + 1)]
    
    # Initialization
    for i in range(n + 1):
        dp[i][0] = True
    # Dynamic programming
    for i in range(1, n + 1):
        for j in range(1, target + 1):
            if j < numbers[i - 1]:
                dp[i][j] = dp[i - 1][j]
            else:
                dp[i][j] = dp[i - 1][j] or dp[i - 1][j - numbers[i - 1]]
                
    return dp[n][target]

class CostConstrainedGroupMarket:
    def __init__(self, 
                 Xown, X, Y, 
                 group_per_variable, price_per_group, budget, 
                 maxiter, epsilon, 
                 theta_0=None,
                 lambda_lasso=0,
                 Ymin=None, Ymax=None,
                 fs_filter='Pearson',
                 fs_alpha = 0.05,
                 gain_with_similar=True):
        self.Xown = Xown
        prices_index = [(X.columns[i], group_per_variable[i], price_per_group[i]) for i in range(len(price_per_group))]
        new_index = pd.MultiIndex.from_tuples(prices_index, names=['variable','group_per_variable', 'price_per_group'])
        X_ = X.copy()
        X_.columns = new_index
        self.X = X_
        self.Y = Y
        self.group_per_variable = pd.DataFrame([group_per_variable], columns = X.columns)
        self.costs_per_group = pd.DataFrame([price_per_group], columns = X.columns)
        possible_prices = np.where([subset_sum(list(price_per_group), b) for b in budget])[0]
        self.budget = np.array(budget)[possible_prices] 
        self.maxiter = maxiter
        self.epsilon = epsilon
        self.theta_0 = theta_0
        self.lambda_lasso = lambda_lasso
        self.Ymin = Ymin
        self.Ymax = Ymax
        self.fs_filter = fs_filter
        self.fs_alpha = fs_alpha
        self.gain_with_similar = gain_with_similar      
        if Xown is None:
            self.own_features = []
        else:
            self.own_features = self.Xown.columns.to_list()

    ### A. HYPERPARAMETER OPTIMIZATION FUNCTIONS
    def spline_model_bopt(self, X, Y, cv_ = 10, n_initial_points = 5, n_calls = 6, delta = 10**(-4)):
        
        spline_lasso_space = [Integer(2, 5, name='degree'), 
                              Integer(2, 10, name='n_knots'), 
                              Real(10**(-7), 10, name='lasso_alpha')]
        
        @use_named_args(spline_lasso_space)
        def objective(**params):
            print(params)
            n_knots = params['n_knots']
            degree = params['degree']
            lasso_alpha = params['lasso_alpha']
            
            if isinstance(cv_, int):
                kf = KFold(n_splits = cv_)
            else: 
                kf = cv_
            rmse_ = 0
            for train_index, test_index in kf.split(self.X):
                Xtr = self.X.iloc[train_index,:]
                Xtr.columns = Xtr.columns.get_level_values(0)
                Ytr = self.Y.iloc[train_index,:]
                Xval = self.X.iloc[test_index,:]
                Xval.columns = Xval.columns.get_level_values(0)
                Yval = self.Y.iloc[test_index,:]
                slasso_model = CustomSplineLassoModelPipeline(self.own_features, 
                                                              n_knots, degree, 
                                                              lasso_alpha, 
                                                              significance_level=self.fs_alpha,
                                                              fs_filter=self.fs_filter)
                slasso_model.fit(Xtr, Ytr)
                ypred = slasso_model.predict(Xval)
                residuals_ = Yval.values - pd.DataFrame(ypred)
                rmse_ += np.sqrt((residuals_**2).mean())[0]
            return rmse_
        
        spline_lasso_res = gp_minimize(objective, 
                                       spline_lasso_space, 
                                       n_calls = n_calls, 
                                       n_initial_points = n_initial_points, 
                                       n_jobs = -1, 
                                       random_state = 321,
                                       callback = DeltaXStopper(delta=delta),                                   
                                       verbose = True)
        opt_params = {spline_lasso_space[i].name: spline_lasso_res.x[i] for i in range(len(spline_lasso_res.x))}
        out_bo = np.hstack((spline_lasso_res.x_iters, spline_lasso_res.func_vals[:,None]))
        return opt_params, out_bo
    
    def tune_models(self, cv_ = 10, n_initial_points = 5, n_calls = 6, delta = 10**(-4)):
        
        print('Optimizing collaborative model hyper-parameters (all data)...')
        spline_lasso_res, xiters = self.spline_model_bopt(self.X, self.Y, cv_, n_initial_points, n_calls, delta)
        self.transformer__degree = spline_lasso_res['degree']
        self.transformer__n_knots = spline_lasso_res['n_knots']
        self.lambda_lasso = spline_lasso_res['lasso_alpha']
        spline_transformer = SplineTransformer(n_knots=self.transformer__n_knots,
                              degree=self.transformer__degree)
        slasso_model = CustomSplineLassoModelPipeline(self.own_features, 
                                                    self.transformer__n_knots, self.transformer__degree, 
                                                    self.lambda_lasso, 
                                                    significance_level=self.fs_alpha,
                                                    fs_filter=self.fs_filter)
        slasso_model.fit(self.X, self.Y)
        # Use ColumnTransformer to apply SplineTransformer to the appropriate columns
        preprocessor = ColumnTransformer(transformers=[('spline', spline_transformer, self.X.columns)])        
        # Create a pipeline with the preprocessor
        pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
        self.spline_transformer = pipeline.fit(self.X)
        self.scaler = StandardScaler().fit(self.spline_transformer.transform(self.X))
        # Transform data according to the results
        X = self.spline_transformer.transform(self.X)   
        X = self.scaler.transform(X)
        self.grid_search_ = xiters
        
        print('Optimizing local model hyper-parameters...')
        own_spline_lasso = CustomSplineLassoModelPipeline(self.own_features, 
                                                    self.transformer__n_knots, self.transformer__degree, 
                                                    self.lambda_lasso, 
                                                    significance_level=self.fs_alpha,
                                                    fs_filter=self.fs_filter)
        own_spline_lasso.fit(self.Xown, self.Y)
        self.own_lasso_model = own_spline_lasso
    
    def tune_per_budget(self, 
                        group_per_variable, 
                        costs_per_group,
                        grid_search_,
                        cv_):
        # Refine hyper-parameters per budget
        if isinstance(cv_, int):
                kf = KFold(n_splits = cv_)
        else: 
            kf = cv_
        rmse_grid = np.hstack((grid_search_.copy(), np.zeros((grid_search_.shape[0],len(self.budget)))))
        unique_splines = np.unique(grid_search_[:,:2], axis=0)
        unique_lasso = np.unique(grid_search_[:,2], axis=0)
        rmse_grid = np.hstack((grid_search_, np.zeros((grid_search_.shape[0],len(self.budget)))))
        for d, k in unique_splines:
            for i, (train_index, test_index) in enumerate(kf.split(self.X)):
                Xtr = self.X.iloc[train_index,:]
                Xtr.columns = Xtr.columns.get_level_values(0)
                Ytr = self.Y.iloc[train_index,:]
                Xval = self.X.iloc[test_index,:]
                Yval = self.Y.iloc[test_index,:]
                # Initialize the market 
                market_ = CostConstrainedGroupMarket(None, Xtr, Ytr,
                                                    group_per_variable, 
                                                    costs_per_group, 
                                                    self.budget,
                                                    self.maxiter, 
                                                    self.epsilon,
                                                    self.theta_0)
                # Fit cost-constrained spline LASSO model
                market_.transformer__degree = int(d)
                market_.transformer__n_knots = int(k)
                for lambda_ in unique_lasso:
                    market_.lambda_lasso = lambda_
                    market_.spline_transformer = SplineTransformer(n_knots=market_.transformer__n_knots,
                                          degree=market_.transformer__degree).fit(market_.X)
                    market_.scaler = StandardScaler().fit(market_.spline_transformer.transform(market_.X))
                    market_.fit_aux()
                    Yvalpred = market_.predict_aux(Xval)
                    Yval_ = pd.concat([Yval]*len(self.budget), axis=1, ignore_index=True)
                    residuals = Yval_-Yvalpred.values
                    rmse_ = np.sqrt((residuals**2).mean(axis=0))
                    index_row = np.where((rmse_grid[:,0]==d) &\
                                         (rmse_grid[:,1]==k) &\
                                         (rmse_grid[:,2]==lambda_))[0]
                    rmse_grid[index_row,3:] = rmse_grid[index_row,3:] + rmse_.values
        if isinstance(cv_, int):
            cv_folds = cv_
        else: 
            cv_folds = len(cv_.unique_folds)
        rmse_grid[:,3:] = rmse_grid[:,3:]/cv_folds        
        rmse_grid = pd.DataFrame(rmse_grid)
        rmse_grid.columns = ['degree', 'nknots', 'alpha'] + [x for x in self.budget]
        self.rmse_grid = rmse_grid
        optimal_per_budget = rmse_grid.iloc[:,:3].iloc[rmse_grid.iloc[:,3:].round(2).idxmin(0),:]
        optimal_per_budget.index = self.budget
        self.optimal_per_budget = optimal_per_budget
        return self 
    
    def cost_constrained_solver(self, X, Y, n, p, C, budg_, theta_0=None):
        if theta_0==None:
            theta_k = pd.DataFrame(np.zeros((X.shape[1],1)), index = X.columns)
        theta_final = np.zeros((X.shape[1], len(budg_)))
        eval_func = np.zeros((len(budg_), self.maxiter))
        for i_b_, b_ in enumerate(self.budget):
            for m in range(self.maxiter):
                a_k = theta_k + np.dot(X.T, (Y - np.dot(X, theta_k))) / (n * C)
                left_part = ((a_k ** 2) - 2 * self.lambda_lasso * np.abs(a_k) + self.lambda_lasso ** 2)
                right_part = ((1 + np.sign(np.abs(a_k) - self.lambda_lasso)) / 4)
                individual_weights = left_part * right_part
                a_k_per_group = individual_weights.groupby(level=['group_per_variable', 'price_per_group']).sum()
                # groups allocation according to budget
                w_k = a_k.copy()
                w_k[:] = 0
                w_k_groups = w_k.index.get_level_values(2)
                indices = knapsack(a_k_per_group.index.get_level_values(1), 
                                   a_k_per_group.values[:,0],                                    
                                   b_)['indices']
                allocated_groups = a_k_per_group.index[indices].get_level_values(0)
                w_k.iloc[w_k_groups.isin(allocated_groups), 0]=1
                theta_k = np.sign(a_k-self.lambda_lasso) * \
                         np.maximum(np.abs(a_k)-self.lambda_lasso, 0) * \
                         w_k
                theta_final[:,i_b_] = theta_k.values[:,0]
                eval_func[i_b_, m] = np.sqrt(np.sum((Y - np.dot(X, theta_k)) ** 2) / n).values#[0]
                if (m>20):
                    if np.all(abs(eval_func[i_b_, m-1] - eval_func[i_b_, m]) <= self.epsilon):
                        break
        if m == self.maxiter:
            print(f"Algorithm does not converge in {self.maxiter} iterations!")
        allocated_vars = np.zeros((theta_final.shape))
        allocated_vars[theta_final!=0] = 1
        for col_ in range(theta_final.shape[1]):
            target_column = allocated_vars[:,col_]
            col_equal_to_target = max(np.where(np.all(allocated_vars == target_column[:, None], axis=0))[0])
            theta_final[:,col_] = theta_final[:,col_equal_to_target]
        theta_final = pd.DataFrame(theta_final)
        theta_final.index = X.columns
        theta_final.columns = budg_
        
        return theta_final, eval_func
    
    def fit_aux(self):        
        # Transform data according to the results    
        # Collab model
        X = self.spline_transformer.transform(self.X.copy())   
        X = self.scaler.transform(X)
        n_splines = self.transformer__n_knots + self.transformer__degree - 1
        index_after_splines = [(f'{name_[0]}_splines{j+1}_', name_[0], name_[1], name_[2]) for name_ in self.X.columns for j in range(n_splines)]
        index_after_splines = [('intercept', 'intercept', 'intercept', 0)] + index_after_splines
        new_index = pd.MultiIndex.from_tuples(index_after_splines, 
                                              names=['spline','variable',
                                                     'group_per_variable', 
                                                     'price_per_group'])
        self.Xsplines = pd.DataFrame(np.concatenate((np.ones((X.shape[0],1)),X),axis=1),
                         columns = new_index)
                
        results = []
        for col in self.Xsplines.columns:
            if col[0]=='intercept':
                correlation, p_value = 1, 0
            else:
                correlation, p_value = pearsonr(self.Xsplines[col], self.Y.values[:,0])
            results.append((correlation, p_value))
        df_corrs = pd.DataFrame(results, columns=['corrs', 'p-value'])
        df_corrs.index = self.Xsplines.columns
        df_corrs['cor_sum_vars'] =  df_corrs['corrs'].abs().groupby(level=['group_per_variable']).transform('sum')
        max_price = self.Xsplines.columns.get_level_values(3).values.max()
        price_normalized = self.Xsplines.columns.get_level_values(3).values/max_price
        df_corrs['ratio_cor_price'] = df_corrs['cor_sum_vars']/price_normalized
        if self.fs_filter=='Pearson':
            df_corrs = df_corrs[df_corrs['p-value']<0.05]
        df_corrs = df_corrs.sort_values(by=['ratio_cor_price'], ascending=False)
        
        X_sorted = self.Xsplines[df_corrs.index].copy()
        
        theta_all = np.zeros((len(self.budget), self.Xsplines.shape[1]))
        theta_all = pd.DataFrame(theta_all)
        theta_all.columns = self.Xsplines.columns
        n, p = self.Xsplines.shape
        L = max(eigh(np.dot(self.Xsplines.T, self.Xsplines) / n)[0]) + 0.1
        theta_k, eval_func = self.cost_constrained_solver(X_sorted, self.Y, n, p, L,
                                                   budg_=self.budget)
        theta_k.columns = self.budget
        theta_all.index = self.budget
        theta_all[theta_k.index] = theta_k.T
        self.estimated_coefs_market = theta_all
        
    def fit(self):        
        
        coefs_per_budget = [pd.DataFrame() for _ in range(self.optimal_per_budget.shape[0])]
        unique_splines = np.unique(self.optimal_per_budget.iloc[:,:2], axis=0)
        spline_transformer_per_budget = [[] for i in range(len(self.budget))]
        spline_scaler_per_budget = [[] for i in range(len(self.budget))]
        for D, K in unique_splines:
            self.transformer__degree = int(D)
            self.transformer__n_knots = int(K)
            spline_transformer = SplineTransformer(n_knots=self.transformer__n_knots,
                                  degree=self.transformer__degree)
            # Use ColumnTransformer to apply SplineTransformer to the appropriate columns
            preprocessor = ColumnTransformer(
                transformers=[('spline', spline_transformer, self.X.columns)]
            )        
            # Create a pipeline with the preprocessor
            pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
            self.spline_transformer = pipeline.fit(self.X)
            self.scaler = StandardScaler().fit(self.spline_transformer.transform(self.X))
            # Transform data according to the results
            X = self.spline_transformer.transform(self.X)   
            X = self.scaler.transform(X)
            # Transform data according to the results: collab model
            X = self.spline_transformer.transform(self.X.copy())   
            X = self.scaler.transform(X)
            n_splines = self.transformer__n_knots + self.transformer__degree - 1
            index_after_splines = [(f'{name_[0]}_splines{j+1}_', name_[0], name_[1], name_[2]) for name_ in self.X.columns for j in range(n_splines)]
            index_after_splines = [('intercept', 'intercept', 'intercept', 0)] + index_after_splines
            own_index_after_splines = [(f'{name_[0]}_splines{j+1}_', name_[0], name_[1], name_[2]) for name_ in self.X.columns if name_[0] in self.Xown.columns for j in range(n_splines)]
            new_index = pd.MultiIndex.from_tuples(index_after_splines, 
                                                  names=['spline','variable',
                                                         'group_per_variable', 
                                                         'price_per_group'])
            self.Xsplines = pd.DataFrame(np.concatenate((np.ones((X.shape[0],1)),X),axis=1),
                             columns = new_index)
                    
            results = []
            for col in self.Xsplines.columns:
                if col[0]=='intercept':
                    correlation, p_value = 1, 0
                else:
                    correlation, p_value = pearsonr(self.Xsplines[col], self.Y.values[:,0])
                results.append((correlation, p_value))
            df_corrs = pd.DataFrame(results, columns=['corrs', 'p-value'])
            df_corrs.index = self.Xsplines.columns
            if self.fs_filter=='Pearson':
                df_corrs = df_corrs[df_corrs['p-value']<self.fs_alpha]
            if self.fs_filter=='PartialPearson':
                df_corrs = df_corrs[df_corrs['p-value']<self.fs_alpha]
                Xothers = self.Xsplines[[x for x in index_after_splines if x not in own_index_after_splines]]
                Xown = self.Xsplines[[x for x in index_after_splines if x in own_index_after_splines]]
                coefs_X = np.linalg.inv(Xown.T@Xown) @ (Xown.T @ Xothers)
                coefs_X.index = Xown.columns
                X_residuals = Xothers - Xown @ coefs_X
                ypred = LinearRegression().fit(Xown, self.Y).predict(Xown)
                Y_residuals =self.Y - ypred                
                for feature in df_corrs.index:
                    if feature not in own_index_after_splines:
                        x_array = X_residuals[[feature]].to_numpy().ravel()
                        y_array = Y_residuals.to_numpy().ravel()
                        partial_corr, p_value = pearsonr(x_array, y_array)
                        df_corrs.loc[feature, 'p-value'] = p_value   
                df_corrs = df_corrs[df_corrs['p-value']<self.fs_alpha]
                
            df_corrs['cor_sum_vars'] =  df_corrs['corrs'].abs().groupby(level=['group_per_variable']).transform('sum')
            max_price = self.Xsplines[df_corrs.index].columns.get_level_values(3).values.max()
            price_normalized = self.Xsplines[df_corrs.index].columns.get_level_values(3).values/max_price
            df_corrs['ratio_cor_price'] = df_corrs['cor_sum_vars']/price_normalized            
            df_corrs = df_corrs.sort_values(by=['ratio_cor_price'], ascending=False)
            
            X_sorted = self.Xsplines[df_corrs.index].copy()
            
            index_row = np.where((self.optimal_per_budget.iloc[:,0]==D) &\
                                (self.optimal_per_budget.iloc[:,1]==K))[0]
            for lambda_ in np.unique(self.optimal_per_budget.iloc[index_row,2]):    
                self.lambda_lasso = lambda_
                theta_all = np.zeros((len(self.budget), self.Xsplines.shape[1]))
                theta_all = pd.DataFrame(theta_all)
                theta_all.columns = self.Xsplines.columns
                # Looping through budget values and performing coefficients' estimation
                n, p = self.Xsplines.shape
                C = max(eigh(np.dot(self.Xsplines.T, self.Xsplines) / n)[0]) + 0.1                
                theta_m, eval_func = self.cost_constrained_solver(X_sorted, self.Y, 
                                                                 n, p, C,
                                                                 budg_=self.budget)
                theta_m.columns = self.budget
                theta_all.index = self.budget
                theta_all[theta_m.index] = theta_m.T
                
                index_budget = np.where((self.optimal_per_budget.iloc[:,0]==D) &\
                                (self.optimal_per_budget.iloc[:,1]==K) &\
                                (self.optimal_per_budget.iloc[:,2]==lambda_))[0]
                for i_budget in index_budget:
                    coefs_per_budget[i_budget] = theta_all.iloc[i_budget,:]
                    spline_transformer_per_budget[i_budget] = self.spline_transformer
                    spline_scaler_per_budget[i_budget] = self.scaler
        
        self.estimated_coefs_market = coefs_per_budget
        self.spline_transformer_per_budget = spline_transformer_per_budget
        self.spline_scaler_per_budget = spline_scaler_per_budget
        
    def predict_aux(self, X):
        X.columns = self.X.columns
        X = self.spline_transformer.transform(X)   
        X = self.scaler.transform(X)
        n_splines = self.transformer__n_knots + self.transformer__degree - 1
        index_after_splines = [(f'{name_[0]}_splines{j+1}_', name_[0], name_[1], name_[2]) for name_ in self.X.columns for j in range(n_splines)]
        index_after_splines = [('intercept', 'intercept', 'intercept', 0)] + index_after_splines
        new_index = pd.MultiIndex.from_tuples(index_after_splines, 
                                              names=['spline','variable',
                                                     'group_per_variable', 
                                                     'price_per_group'])
        X = pd.DataFrame(np.concatenate((np.ones((X.shape[0],1)),X),axis=1),
                         columns = new_index)
        Yhat = X @ self.estimated_coefs_market.T
        return Yhat

    def predict(self, X):
        Yhat = pd.DataFrame(np.zeros((X.shape[0], len(self.budget))))
        X.columns = self.X.columns
        for i_b_, b_ in enumerate(self.budget):
            X_ = self.spline_transformer_per_budget[i_b_].transform(X)   
            X_ = self.spline_scaler_per_budget[i_b_].transform(X_)
            n_splines = self.transformer__n_knots + self.transformer__degree - 1
            index_after_splines = [(f'{name_[0]}_splines{j+1}_', name_[0], name_[1], name_[2]) for name_ in self.X.columns for j in range(n_splines)]
            index_after_splines = [('intercept', 'intercept', 'intercept', 0)] + index_after_splines
            new_index = pd.MultiIndex.from_tuples(index_after_splines, 
                                                  names=['spline','variable','group_per_variable', 'price_per_group'])
            X_ = pd.DataFrame(np.concatenate((np.ones((X_.shape[0],1)),X_),axis=1),
                             columns = new_index)
            Yhat.iloc[:, i_b_] = X_ @ self.estimated_coefs_market[i_b_]
        return Yhat      
    
    def BidGainTable(self, Xown, X, Y, k=100):
        
        if self.gain_with_similar: # estimate gains based on similar timestamps
            Yownhat = self.own_lasso_model.predict(self.Xown)
            if self.Ymin!=None: 
                Yownhat[Yownhat<self.Ymin] = self.Ymin
            if self.Ymax!=None: 
                Yownhat[Yownhat>self.Ymax] = self.Ymax
            residuals_local = self.Y.values[:,0] - Yownhat 
            # Market model: error estimation
            bgt = np.zeros((len(self.budget), X.shape[0]))
            X.columns=self.X.columns            
            residuals_colab = self.Y.values - self.predict(self.X)
            for i_b_, b_ in enumerate(self.budget):
                Xsplines_tr = self.spline_transformer_per_budget[i_b_].transform(self.X)
                Xsplines_tr = self.spline_scaler_per_budget[i_b_].transform(Xsplines_tr)
                Xsplines_ts = self.spline_transformer_per_budget[i_b_].transform(X)
                Xsplines_ts = self.spline_scaler_per_budget[i_b_].transform(Xsplines_ts)
                distances = cdist(Xsplines_ts, Xsplines_tr, metric='euclidean')
                ordered_obs = np.argsort(distances, axis=1)[:,1:(k+1)]
                rmse_local = np.sqrt(np.mean(residuals_local[ordered_obs]**2, axis=1))
                rmse_colab = np.sqrt(np.mean(residuals_colab.iloc[ordered_obs[0], i_b_]**2, axis=0))
                improvement = 100*(rmse_local-rmse_colab)/rmse_local
                improvement[improvement<0] = 0
                bgt[i_b_, :] = improvement
            Yhat_ts = self.predict(X)            
            bgt = pd.DataFrame(bgt)
            bgt.columns = ['gain']#X.index
            bgt.insert(loc=0, column='bid', value=self.budget)            
            self.bgt = bgt.round(1)
            self.forecasts_per_budget = Yhat_ts.T            
        else: # estimate gains based on validation set
            Yownhat = self.own_lasso_model.predict(Xown)
            if self.Ymin!=None: 
                Yownhat[Yownhat<self.Ymin] = self.Ymin
            if self.Ymax!=None: 
                Yownhat[Yownhat>self.Ymax] = self.Ymax            
            rmse_local = np.sqrt(np.mean((Y.values[:,0] - Yownhat)**2, axis=0))        
            # Market model: error estimation
            Yhat = self.predict(X)            
            if self.Ymin!=None: 
                Yhat[Yhat<self.Ymin] = self.Ymin
            if self.Ymax!=None: 
                Yhat[Yhat>self.Ymax] = self.Ymax
            rmse_per_budget = np.sqrt(np.mean((Y.values - Yhat)**2, axis=0))
            # Bid-Gain-Table: 
            improvement = 100*(rmse_local-rmse_per_budget)/rmse_local
            improvement[improvement<0] = 0
            bgt_ = pd.DataFrame({'bid':[0], 'gain': [0]})
            bgt = pd.DataFrame({'bid':self.budget, 'gain': improvement})
            bgt = pd.concat((bgt_, bgt))
            self.bgt = bgt.round(1)
            
    def price(self, ValueFunction):
        self.bgt['VF'] = ValueFunction(self.bgt[['gain']])
        x, y = sp.symbols('x y')   
        vf_constant = sp.simplify(ValueFunction(x)).is_constant() 
        if vf_constant:
            acceptable_prices = self.bgt.loc[(self.bgt['bid'] <= ValueFunction(1))]
        else:
            acceptable_prices = self.bgt.loc[(self.bgt['VF'] >= self.bgt['bid'])]
        
        if acceptable_prices.empty:
            return self
            
        if self.gain_with_similar:
            minimum_pos = acceptable_prices.iloc[:, 1:-1].idxmax(axis=0)
            final_prices = acceptable_prices['bid'].loc[minimum_pos]
            final_prices.index = acceptable_prices.columns[1:-1]
            final_forecasts = np.zeros(final_prices.shape)
            for obs_ in range(final_prices.shape[0]):
                final_forecasts[obs_] = self.forecasts_per_budget.iloc[minimum_pos.iloc[obs_], obs_]
            final_forecasts = pd.DataFrame(final_forecasts)
            final_forecasts.index = acceptable_prices.iloc[:, 1:-1].columns
            self.final_price = final_prices
            self.final_forecasts = final_forecasts
        else:
            final_price_candidates = acceptable_prices.loc[acceptable_prices['gain']==acceptable_prices['gain'].max(),'bid']
            
            if not final_price_candidates.empty and all(final_price_candidates!=0):
                self.final_price = final_price_candidates.values[0]
                pos_final_price = np.where(self.budget == self.final_price)[0][0]
                self.final_model_coefs = self.estimated_coefs_market[pos_final_price]
            else:
                self.final_price = None
        
    def forecasting_gain(self, Xown, X, Y):
        X.columns = self.X.columns
        Yownhat = self.own_lasso_model.predict(Xown)
        if self.Ymin!=None: 
            Yownhat[Yownhat<self.Ymin] = self.Ymin
        if self.Ymax!=None: 
            Yownhat[Yownhat>self.Ymax] = self.Ymax
        rmse_local = np.sqrt(np.mean((Y.values[:,0] - Yownhat)**2, axis=0))      
        Yhat = self.predict(X)
        
        if self.Ymin!=None: 
            Yhat[Yhat<self.Ymin] = self.Ymin
        if self.Ymax!=None: 
            Yhat[Yhat>self.Ymax] = self.Ymax
        rmse_per_budget = np.sqrt(np.mean((Y.values - Yhat)**2, axis=0))
        # Bid-Gain-Table: 
        improvement = 100*(rmse_local-rmse_per_budget)/rmse_local
        bgt = pd.DataFrame({'bid':self.budget, 
                            'gain': improvement.round(1), 
                            'local_rmse':rmse_local, 
                            'collab_rmse':rmse_per_budget})
        return bgt
