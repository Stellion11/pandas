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
from sklearn.covariance import EllipticEnvelope
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

# 이상치 탐지 모델 정의
outlier_detector = EllipticEnvelope(contamination=0.01)  # 이상치 비율 설정
outlier_detector.fit(x)

# 이상치 탐지 결과
outlier_preds = outlier_detector.predict(x)  # 1: 정상, -1: 이상치

# 정상 데이터만 선택
x = x[outlier_preds == 1]
y = y[outlier_preds == 1]

print(f"정상 데이터 수: {x.shape[0]} / 전체: {len(outlier_preds)}")

exit()

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


