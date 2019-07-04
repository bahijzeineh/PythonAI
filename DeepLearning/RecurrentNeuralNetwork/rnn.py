# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 13:34:50 2019

@author: Bahij
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')

training_set = dataset_train.iloc[:,1:2].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set_scaled = sc.fit_transform(training_set)

X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
    
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import Dropout

model = Sequential()

model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
model.add(Dropout(rate = 0.2))

model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(rate = 0.2))

model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(rate = 0.2))

model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(rate = 0.2))

model.add(Flatten())

model.add(Dense(units = 1))

model.compile(optimizer = 'adam', loss = 'mean_squared_error')

#model.fit(x=X_train, y=y_train, epochs = 100, batch_size = 32)
model.load_weights('rnn-ep100.mdl')


#predict and compare to actual
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_price = dataset_test.iloc[:,1:2].values

dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test)-60:].values
inputs = inputs.reshape(-1,1)
inputs_scaled = sc.transform(inputs)

#test data must be same input shape as train data so we recreate the structure
X_test = []
for i in range(60, 80):
    X_test.append(inputs_scaled[i-60:i, 0])
    
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_price = model.predict(X_test)

predicted_price = sc.inverse_transform(predicted_price)


#plot data
plt.plot(real_price, color = 'red', label = 'Real price')
plt.plot(predicted_price, color = 'blue', label = 'Predicted price')
plt.title('Google stock prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
