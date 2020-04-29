import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset_train = pd.read_csv('Google_Stock_Price_train.csv')
training_set = dataset_train.iloc[:,1:2].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)

X_train = []
y_train = []
for i in range(20,1258):
    X_train.append(training_set_scaled[i-20:i,0])
    y_train.append(training_set_scaled[i,0])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

import keras 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

regressor = Sequential()

regressor.add(LSTM(units = 3, return_sequences=True, input_shape=(None,1)))
regressor.add(LSTM(units = 3, return_sequences = True))
regressor.add(LSTM(units = 3, return_sequences=True))
regressor.add(LSTM(units = 3))

regressor.add(Dense(units=1))

regressor.compile(optimizer='rmsprop', loss='mean_squared_error')
regressor.fit(X_train, y_train, batch_size=32,epochs = 100)


dataset_test = pd.read_csv('Google_Stock_Price_test.csv')
test_set = dataset_test.iloc[:,1:2].values
real_stock_price = np.concatenate((training_set[0:1258],test_set), axis = 0)


scaled_real_stock_price = sc.fit_transform(real_stock_price)
inputs = []
for i in range(1258,1278):
    inputs.append(scaled_real_stock_price[i-20:i,0])
inputs = np.array(inputs)
inputs = np.reshape(inputs, (inputs.shape[0], inputs.shape[1], 1))
predicted_stock_price = regressor.predict(inputs)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

plt.plot(real_stock_price[1258:], color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
