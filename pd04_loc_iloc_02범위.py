import pandas as pd

data = [
    ['삼성', '1000', '2000'],
    ['현대', '1100', '3000'],
    ['LG', '2000', '500'],
    ['아모레', '3500', '6000'],
    ['네이버', '100', '1500'],
]

index = ['031', '059', '033', '045', '023']
columns = ['종목명', '시가', '종가']

df = pd.DataFrame(data=data, index=index, columns=columns)

print(df.iloc[3][1], df.iloc[4][1])



########## 범위(아모레아와 네이버의 시가) ############
print(df.iloc[3:5]['시가'])
print(df.iloc[3:5, 1])

print(df.loc[['045', '023']]['시가'])
print(df.loc[['045', '023'], '시가'])

print(df.loc['045':]['시가'])
print(df.loc['045':, '시가'])