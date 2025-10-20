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

imputer4 = SimpleImputer(strategy='most_frequent')
data5 = imputer4.fit_transform(data)
# print(data5)
# [[ 2.  2.  2.  8.]
#  [ 8.  4.  4.  4.]
#  [ 6.  4.  6.  8.]
#  [ 8.  8.  8.  8.]
#  [10.  4. 10.  8.]
#  [ 8.  4. 12.  8.]]
imputer5 = SimpleImputer(strategy='constant', fill_value=777)
data6 = imputer5.fit_transform(data)
# print(data6)
# [[  2.   2.   2. 777.]
#  [777.   4.   4.   4.]
#  [  6. 777.   6. 777.]
#  [  8.   8.   8.   8.]
#  [ 10. 777.  10. 777.]
#  [  8.   4.  12.   8.]]
################################################################################
imputer6 = KNNImputer() # KNN알고리즘으로 결측치 처리
data7 = imputer6.fit_transform(data)
# print(data7)
# [[ 2.          2.          2.          6.66666667]
#  [ 6.8         4.          4.          4.        ]
#  [ 6.          4.5         6.          6.66666667]
#  [ 8.          8.          8.          8.        ]
#  [10.          4.5        10.          6.66666667]
#  [ 8.          4.         12.          8.        ]]
################################################################################
imputer7 = IterativeImputer() # 디폴트 : BayesianRide 회귀모델
data8 = imputer7.fit_transform(data)
# print(data8)
# [[ 2.          2.          2.          1.99812493]
#  [ 4.00375533  4.          4.          4.        ]
#  [ 6.          5.98870972  6.          5.99560853]
#  [ 8.          8.          8.          8.        ]
#  [10.          9.98722604 10.          9.99636304]
#  [ 8.          4.         12.          8.        ]]