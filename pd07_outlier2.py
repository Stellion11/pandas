import numpy as np 
aaa = np.array([[-10,2,3,4,5,6,7,8,9,10,11,12,50],
               [100,200,-30,400,500,600,-70000,800,900,1000,210,420,350]]).T

print(aaa, aaa.shape) #(13, 2)

# 이상치 찾기 함수
def outlier(data):
    outliers = []
    for col in range(data.shape[1]):
        col_data = data[:, col]
        q1 = np.percentile(col_data, 25)
        q2 = np.percentile(col_data, 50)
        q3 = np.percentile(col_data, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        print(f"\nColumn {col}:")
        print("Q1:", q1)
        print("Q2 (median):", q2)
        print("Q3:", q3)
        print("IQR:", iqr)
        print("Lower Bound:", lower_bound)
        print("Upper Bound:", upper_bound)

        # 이상치 인덱스 찾기
        outlier_idx = np.where((col_data < lower_bound) | (col_data > upper_bound))[0]
        print("Outlier Indices:", outlier_idx)
        outliers.append(outlier_idx)
    
    return outliers
'''
Column 0:
Q1: 4.0
Q2 (median): 7.0
Q3: 10.0
IQR: 6.0
Lower Bound: -5.0
Upper Bound: 19.0
Outlier Indices: [ 0 12]

Column 1:
Q1: 200.0
Q2 (median): 400.0
Q3: 600.0
IQR: 400.0
Lower Bound: -400.0
Upper Bound: 1200.0
Outlier Indices: [6]
'''

# 실행
outlier_indices = outlier(aaa)

# outlier_indices는 각 열에 대한 이상치 인덱스를 담은 리스트
for col in range(aaa.shape[1]):
    print(f"\nColumn {col} 이상치 값들:")
    col_data = aaa[:, col]
    for idx in outlier_indices[col]:
        print(f"Index {idx}: {col_data[idx]}")
    
# Column 0 이상치 값들:
# Index 0: -10
# Index 12: 50

# Column 1 이상치 값들:
# Index 6: -70000