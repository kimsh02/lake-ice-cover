import pandas as pd;
import matplotlib.pyplot as plt
import numpy as np

# load in csv data
col = 'Days of Ice Cover'
mendota_df = pd.read_csv('mendota.csv').loc[5:175].dropna(subset=[col]).iloc[::-1].reset_index(drop=True)
monona_df = pd.read_csv('monona.csv').loc[6:176].dropna(subset=[col]).iloc[::-1].reset_index(drop=True)

# 3a
# plt.figure(1)
# plt.plot([x for x in range(1855,2019)],mendota_df[col], color='red')
# plt.plot([x for x in range(1855,2019)],monona_df[col], color='green')
# plt.xlabel('Year')
# plt.ylabel(col)
# plt.title('Annual Days of Ice Cover for Two Lakes')
# plt.legend(['Mendota', 'Monona'])
# plt.savefig('ice_cover.png')
# plt.show()

# diff = []
# for i,j in zip(monona_df[col],mendota_df[col]):
#     diff.append(i-j)

# plt.figure(2)
# plt.plot([x for x in range(1855,2019)],diff, color='blue')
# plt.xlabel('Year')
# plt.ylabel('Ice Days_Monona - Ice Days_Mendota')
# plt.title('Difference in Annual Days of Ice Cover between Two Lakes')
# plt.savefig('diff.png')
# plt.show()

# # 3b

mendota_df['Winter'] = mendota_df['Winter'].apply(lambda x: int(x[:4]))
monona_df['Winter'] = monona_df['Winter'].apply(lambda x: int(x[:4]))

split = mendota_df.index[mendota_df['Winter'] == 1970].tolist()[0] + 1
mendota_df_train = mendota_df.iloc[:split]
mendota_df_test = mendota_df.iloc[split:]
split = monona_df.index[monona_df['Winter'] == 1970].tolist()[0] + 1
monona_df_train = monona_df.iloc[:split]
monona_df_test = monona_df.iloc[split:]

# mendota_a = np.array(mendota_df_train[col])
# monona_a = np.array(monona_df_train[col])

# print('Mendota Mean:')
# print(np.mean(mendota_a))
# print('Mendota STD:')
# print(np.std(mendota_a,ddof=1))
# print()
# print('Monona Mean:')
# print(np.mean(monona_a))
# print('Monona STD:')
# print(np.std(monona_a,ddof=1))

# 3c

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

lr = LinearRegression()
monona_df_train['ones'] = 1
monona_df_test['ones'] = 1
lr.fit(monona_df_train[['ones','Winter',col]], mendota_df_train[[col]])
mendota_pred = lr.predict(monona_df_test[['ones','Winter', col]])

weights = lr.coef_
print('Feature weights: ' + str(weights))
intercept = lr.intercept_
print('Intercept:' + str(intercept))

# 3d
mse = mean_squared_error(mendota_df_test[[col]], mendota_pred)
print('Mean sqaure error: ' + str(mse))

# 3e

lr2 = LinearRegression()
lr2.fit(monona_df_train[['ones', 'Winter']],mendota_df_train[[col]])
mendota_pred2 = lr2.predict(monona_df_test[['ones', 'Winter']])
mse2 = mean_squared_error(mendota_df_test[[col]], mendota_pred2)

weights2 = lr2.coef_
intercept2 = lr2.intercept_

print('Feature weights: ' + str(weights2))
print('Intercept:' + str(intercept2))
print('Mean sqaure error: ' + str(mse2))