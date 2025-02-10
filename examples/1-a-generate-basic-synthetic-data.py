import numpy as np
import pandas as pd
from sklearn.utils import shuffle

import os
if not os.path.exists('../data'):
    os.makedirs('../data')

np.random.seed(1)
T = 10000 # nr observations
n = 100 # covariates

X = np.random.normal(0, 1, size=(T, n))
B = np.zeros((n, 1))

non_zero_index = [2, 6, 11, 20, 30, 36, 47, 50, 62, 89]

np.random.seed(2)
B[non_zero_index] = np.array([6, 5, 2, 10, 1, 4, 7, 3, 8,9]).reshape(-1, 1)
np.random.seed(1)
Y1 = X@B + np.random.normal(0, 1, size=(T, 1))
np.random.seed(1)
Y2 = np.exp(0.05 * X@B) + np.random.normal(0, 1, size=(T, 1))


# Writing CSV files for linear case
df_proprio_linear = pd.DataFrame(X[:, :10], columns=[f"X_{i}" for i in range(1, 11)])
df_outros_linear = pd.DataFrame(X[:, 10:], columns=[f"X_{i}" for i in range(11, 101)])
df_y_buyer_linear = pd.DataFrame({'Y': Y1[:,0]})

df_proprio_non_linear = pd.DataFrame(X[:, :10], columns=[f"X_{i}" for i in range(1, 11)])
df_outros_non_linear = pd.DataFrame(X[:, 10:], columns=[f"X_{i}" for i in range(11, 101)])
df_y_buyer_non_linear = pd.DataFrame({'Y': Y2[:,0]})

# Adding columns X_73 and X_74 to df_outros_linear based on df_proprio_linear
np.random.seed(2)
df_outros_linear['X_73'] = df_proprio_linear[['X_3']] + pd.DataFrame({'X_3': np.random.normal(0, 0.01, size=(T, ))})
np.random.seed(3)
df_outros_linear['X_74'] = df_outros_linear[['X_37']] + pd.DataFrame({'X_37': np.random.normal(0, 0.01, size=(T, ))})

df_proprio_linear.to_csv('../data/X_buyer-linear.csv', index=False)
df_outros_linear.to_csv('../data/X_sellers-linear.csv', index=False)
df_y_buyer_linear.to_csv('../data/Y_buyer-linear.csv', index=False)

# Adding columns X_73 and X_74 to df_outros_non_linear based on df_proprio_non_linear
np.random.seed(2)
df_outros_non_linear['X_73'] = df_proprio_non_linear[['X_3']] + pd.DataFrame({'X_3': np.random.normal(0, 0.01, size=(T, ))})
np.random.seed(3)
df_outros_non_linear['X_74'] = df_outros_non_linear[['X_37']] + pd.DataFrame({'X_37': np.random.normal(0, 0.01, size=(T, ))})

df_proprio_non_linear.to_csv('../data/X_buyer-non-linear.csv', index=False)
df_outros_non_linear.to_csv('../data/X_sellers-non-linear.csv', index=False)
df_y_buyer_non_linear.to_csv('../data/Y_buyer-non-linear.csv', index=False)