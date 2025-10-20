import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
################################################################################
data = pd.DataFrame([[2, np.nan, 6, 8, 10, 8],
                     [2, 4, np.nan, 8, np.nan, 4],
                     [2, 4, 6, 8, 10, 12],
                     [np.nan, 4, np.nan, 8, np.nan, 8]
                     ]).T
# print(data)

data.columns = ['x1', 'x2', 'x3', 'x4']
# pip install impyute
from impyute.imputation.cs import mice
data9 = mice(data.values,
            #  n=10,
             seed=777,
             )
print(data9)