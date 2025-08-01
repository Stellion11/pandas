# PCA, FI 사용말고 PF 이용하여 단순 증폭
import numpy as np
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import time

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.linear_model import LinearRegression



from xgboost import XGBClassifier, XGBRegressor

seed = 123
random.seed(seed)
np.random.seed(seed)

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) #(20640, 8) (20640,)

print(datasets.feature_names)
# ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']

# 1. 이상치 시각적으로 확인
def plot_boxplot(data):
    plt.boxplot(data)
    plt.title("Boxplot of all columns")
    plt.xlabel("Feature Index")
    plt.ylabel("Value")
    plt.show()
    
# plot_boxplot(x)
#column 5, 6에 이상치 존재

def detect_outliers_all(data, feature_names=None, iqr_scale=1.5):
    """
    모든 column에 대해 이상치 탐지 (IQR 기반)
    - data: 2D numpy array
    - feature_names: column 이름 리스트 (없으면 번호로 처리)
    - iqr_scale: IQR 기준 범위 조절 (기본값: 1.5)
    """
    n_cols = data.shape[1]
    outlier_results = []

    for col in range(n_cols):
        col_data = data[:, col]
        q1 = np.percentile(col_data, 25)
        q2 = np.percentile(col_data, 50)
        q3 = np.percentile(col_data, 75)
        iqr = q3 - q1
        lower_bound = q1 - iqr_scale * iqr
        upper_bound = q3 + iqr_scale * iqr

        outlier_idx = np.where((col_data < lower_bound) | (col_data > upper_bound))[0]

        if len(outlier_idx) > 0:
            col_name = feature_names[col] if feature_names else f"Column {col}"
            print(f"\n📌 {col_name}")
            print(f"  Q1 = {q1:.4f}, Q3 = {q3:.4f}, IQR = {iqr:.4f}")
            print(f"  이상치 범위: < {lower_bound:.4f} 또는 > {upper_bound:.4f}")
            print(f"  이상치 인덱스 수: {len(outlier_idx)}")
            print(f"  이상치 값 예시 (최대 10개): {col_data[outlier_idx[:10]]}")

            outlier_results.append({
                'column': col_name,
                'indices': outlier_idx,
                'values': col_data[outlier_idx]
            })
    
    if not outlier_results:
        print("🎉 이상치 없음 (모든 컬럼)")
    
    return outlier_results

detect_outliers_all(x)

#이상치 처리
log_cols = ['MedInc', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']
log_indices = [datasets.feature_names.index(col) for col in log_cols]

x_log = x.copy()
for idx in log_indices:
    x_log[:, idx] = np.log1p(x_log[:, idx])

x_train, x_test, y_train, y_test = train_test_split(
    x_log, y, train_size=0.8, random_state=seed,
    # stratify=y
)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
model = RandomForestRegressor(random_state=seed)

start = time.time()
model.fit(x_train, y_train)
end = time.time()

#4. 평가, 예측
y_pred = model.predict(x_test)

#평가
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("R²:", r2)
print("RMSE:", rmse)
print("걸린시간:", round(end - start, 2), '초')

# R²: 0.8050400211005855
# RMSE: 0.509176333292196
# 걸린시간: 70.71 초

# 개선있음
# R²: 0.8123733510748883
# RMSE: 0.49950833068872647
# 걸린시간: 11.78 초


