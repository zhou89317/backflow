import pandas as pd
from sktime.forecasting.model_selection import temporal_train_test_split

df = pd.read_csv('df.csv', index_col='Date',parse_dates=['Date'])
df = df.iloc[::-1]

X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = temporal_train_test_split(X, y, test_size=0.2)
print(X_train,X_train.shape)
print((X_test,X_test.shape))
print(y_train, y_train.shape)
print(y_test,y_test.shape)