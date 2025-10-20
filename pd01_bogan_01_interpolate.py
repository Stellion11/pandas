"""
결측치 처리
1. 삭제 - 행 또는 열
2. 임의의 값
    - 0 : fillna
    - 평균 : mean (이상치의 문제가 있음)
    - 중위 : median
    - 앞값 : ffill
    - 뒷값 : bfill
    - 특정값 : 777 (조건 보고 넣는게 낫다)
    - 기타...
3. interpolate - 보간(알려진 데이터 점 집합의 범위 내에 새 데이터 점을 추가하는 기법)
4. 모델 : .predict(값을 예측해서), (전혀 다른 모델 사용)
5. 부스팅 계열 모델 : 통상 이상치, 결측치에 대해 영향을 덜 받는다(자유롭다)
"""
import pandas as pd
import numpy as np
import datetime

dates = [
    '16/7/2025',
    '17/7/2025',
    '18/7/2025',
    '19/7/2025',
    '20/7/2025',
    '21/7/2025',
]

dates = pd.to_datetime(dates)
print(dates)

print("==============================")
ts = pd.Series([2, np.nan, np.nan, 8, 10, np.nan], index=dates)
print(ts)

print("==============================")
ts = ts.interpolate()
print(ts)

ts = pd.Series([2, 3, 4, 5, 6, 7], index=dates)
print(ts)

# 중간값들은 linear로 채워짐. 마지막은 ffill (이전값으로 채워짐)