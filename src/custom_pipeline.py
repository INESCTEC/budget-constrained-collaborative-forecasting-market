import numpy as np
import pandas as pd
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.linear_model import Lasso, LinearRegression
from scipy.stats import pearsonr
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class CustomSplineLassoModelPipeline:
    def __init__(self, agent_features, n_knots, degree, lasso_alpha, significance_level=0.05, fs_filter='Pearson'):
        self.agent_features = agent_features  # List of features to control for
        self.scaler = StandardScaler()
        self.selected_features_mask = None
        self.n_knots = n_knots
        self.degree = degree
        self.lasso_alpha = lasso_alpha
        self.significance_level = significance_level
        self.fs_filter = fs_filter
        
    def fit(self, X, y):
        """Fit the model using spline transformation, partial correlation feature selection, and Lasso."""
        # 1. Apply spline transformation
        self.spline_transformer = SplineTransformer(n_knots=self.n_knots, degree=self.degree)
        #preprocessor = ColumnTransformer(transformers=[('spline', spline_transformer, X.columns)])        
        # Create a pipeline with the preprocessor
        #pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
        #self.spline_transformer = pipeline.fit(X)
        splines_per_feature = self.n_knots + self.degree - 1
        splines_names = [f'{x}_s{s_}_' for x in X.columns.get_level_values(0) for s_ in range(1, splines_per_feature+1)]
        X_splines = pd.DataFrame(self.spline_transformer.fit_transform(X), columns = splines_names)
        self.own_splines = [f'{x}_s{s_}_' for x in self.agent_features for s_ in range(1, splines_per_feature+1)]
        # 2. Feature selection using partial Pearson correlation
        if self.fs_filter=='PartialPearson': 
            self._select_features(X_splines, y)
            if len(self.own_splines)>0:
                self._select_features_partial(X_splines[self.selected_features_], y)
        if self.fs_filter=='Pearson': 
            self._select_features(X_splines, y)
        if self.fs_filter is None: 
            self.selected_features_ = X_splines.columns.to_list()
        X_selected = X_splines[self.selected_features_]        
        # 3. Scale the selected features
        X_scaled = self.scaler.fit_transform(X_selected)
        # 4. Fit the Lasso model
        self.model = Lasso(alpha=self.lasso_alpha)
        self.model.fit(X_scaled, y)

    def predict(self, X):
        """Make predictions using the fitted model."""
        # Apply the same transformation steps
        splines_per_feature = self.n_knots + self.degree - 1
        splines_names = [f'{x}_s{s_}_' for x in X.columns.get_level_values(0) for s_ in range(1, splines_per_feature+1)]
        X_splines = pd.DataFrame(self.spline_transformer.transform(X), columns = splines_names)
        X_selected = X_splines[self.selected_features_]
        X_scaled = self.scaler.transform(X_selected)        
        # Predict using the fitted Lasso model
        return self.model.predict(X_scaled)
    
    def _partial_corr(self, x, y, z):
        """Compute partial correlation between x and y, controlling for z (other features)."""
        # Regress x and y on z
        x_residuals = x - LinearRegression().fit(z, x).predict(z)
        y_residuals = y - LinearRegression().fit(z, y).predict(z)
        # Compute Pearson correlation on residuals
        return pearsonr(x_residuals, y_residuals)

    def _select_features_partial(self, X, y):
        """Select features based on partial Pearson correlation."""
        # Convert to DataFrame for easier manipulation
        Xothers = X[[x for x in X if x not in self.own_splines]]
        Xown = X[[x for x in X if x in self.own_splines]]
        coefs_X = np.linalg.inv(Xown.T@Xown) @ (Xown.T @ Xothers)
        coefs_X.index = Xown.columns
        X_residuals = Xothers - Xown @ coefs_X
        ypred = LinearRegression().fit(Xown, y).predict(Xown)
        Y_residuals =y - ypred
        
        self.selected_features_ = []
        for feature in X.columns:
            if feature not in [x for x in self.own_splines if x in X.columns]:
                # Compute partial correlation of each feature with target, controlling for agent_features
                x_array = X_residuals[[feature]].to_numpy().ravel() 
                y_array = Y_residuals.to_numpy().ravel()
                partial_corr, p_value = pearsonr(x_array, y_array)
            else:
                x_array = X[[feature]].to_numpy().ravel()  
                y_array = y.to_numpy().ravel()
                partial_corr, p_value = pearsonr(x_array, y_array)
            if p_value < self.significance_level:
                self.selected_features_.append(feature)

    def _select_features(self, X, y):
        """Select features based on partial Pearson correlation."""
        # Convert to DataFrame for easier manipulation
        self.selected_features_ = []
        for feature in X.columns:
            corr_, p_value = pearsonr(X[[feature]].to_numpy().ravel(), y.to_numpy().ravel())
            if p_value < self.significance_level:
                self.selected_features_.append(feature)
