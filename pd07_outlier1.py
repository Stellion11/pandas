import numpy as np 
aaa = np.array([-10,2,3,4,5,6,7,8,9,10,11,12,50])

def outlier(data):
    q1, q2, q3 = np.percentile(data, [25,50,75])
    print('1사분위 : ', q1)
    print('2사분위 : ', q2)
    print('3사분위 : ', q3)
    iqr = q3 - q1
    print('IQR : ', iqr)
    lower_bound = q1 - (iqr*1.5)
    upper_bound = q3 + (iqr*1.5)
    return np.where((data > upper_bound) | (data < lower_bound)), \
        iqr, lower_bound, upper_bound

outlier_loc, iqr, low, up = outlier(aaa)
print('이상치의 위치 : ', outlier_loc)
# 1사분위 :  4.0
# 2사분위 :  7.0
# 3사분위 :  10.0
# IQR :  6.0
# 이상치의 위치 :  (array([ 0, 12], dtype=int64),)

import matplotlib.pyplot as plt
plt.boxplot(aaa)
plt.axhline(up, color='pink', label='upper bound')
plt.axhline(low, color='pink', label='lower bound')
plt.legend()
plt.show()
