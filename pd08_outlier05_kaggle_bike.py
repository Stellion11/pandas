# PCA, FI 사용말고 PF 이용하여 단순 증폭
import numpy as np
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import time

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


from xgboost import XGBClassifier, XGBRegressor

seed = 123
random.seed(seed)
np.random.seed(seed)

# 1. 데이터
path = './_data/kaggle/bike/'           # 상대경로 : 대소문자 구분X

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sampleSubmission.csv')

print(train_csv.columns)
# ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
#        'humidity', 'windspeed', 'casual', 'registered', 'count']

x = train_csv.drop(['count'], axis=1)
y = train_csv['count']

# 1. 이상치 시각적으로 확인
def plot_boxplot(data):
    plt.boxplot(data)
    plt.title("Boxplot of all columns")
    plt.xlabel("Feature Index")
    plt.ylabel("Value")
    plt.show()

# plot_boxplot(x)

def detect_outliers_all(data, feature_names=None, iqr_scale=1.5):
    data_np = data.values  # 🔸 DataFrame → numpy 변환
    n_cols = data_np.shape[1]
    outlier_results = []

    for col in range(n_cols):
        col_data = data_np[:, col]
        q1 = np.percentile(col_data, 25)
        q2 = np.percentile(col_data, 50)
        q3 = np.percentile(col_data, 75)
        iqr = q3 - q1
        lower_bound = q1 - iqr_scale * iqr
        upper_bound = q3 + iqr_scale * iqr

        outlier_idx = np.where((col_data < lower_bound) | (col_data > upper_bound))[0]

        if len(outlier_idx) > 0:
            col_name = feature_names[col] if feature_names else data.columns[col]
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
'''
📌 humidity
  Q1 = 47.0000, Q3 = 77.0000, IQR = 30.0000
  이상치 범위: < 2.0000 또는 > 122.0000
  이상치 인덱스 수: 22
  이상치 값 예시 (최대 10개): [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

📌 windspeed
  Q1 = 7.0015, Q3 = 16.9979, IQR = 9.9964
  이상치 범위: < -7.9931 또는 > 31.9925
  이상치 인덱스 수: 227
  이상치 값 예시 (최대 10개): [32.9975 36.9974 35.0008 35.0008 39.0007 35.0008 35.0008 36.9974 32.9975
 36.9974]

'''
# 위에 두 컬럼만 이상치 처리

####### x와 y 분리 ######
x = train_csv.drop(['casual', 'registered', 'count'], axis=1)    # test셋에는 없는 casual, registered, y가 될 count는 x에서 제거
# y = train_csv['count'] # 위에서 이미 해줌

# 이상치_humidity
x.loc[x['humidity'] == 0, 'humidity'] = np.nan
x['humidity'] = x['humidity'].interpolate()  # 선형 보간

# 이상치_windspeed
# 평균/보간 등으로 대체
x.loc[x['windspeed'] > 31.9925, 'windspeed'] = np.nan
x['windspeed'] = x['windspeed'].interpolate()

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=seed,
    # stratify=y
)

scaler = StandardScaler()
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

# R²: 0.2772909613252007
# RMSE: 152.08695010773602
# 걸린시간: 9.4 초

# R²: 0.2661665541443171
# RMSE: 153.2529907149747
# 걸린시간: 1.68 초