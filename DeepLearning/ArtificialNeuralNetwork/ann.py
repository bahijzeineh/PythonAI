# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 15:58:28 2019

@author: Bahij
"""

#data preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Value Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#The ANN

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

model = Sequential()

#add input and first hidden layer with dropout
model.add(Dense(activation = 'relu', units = 6, kernel_initializer = 'uniform', input_dim = 10))
model.add(Dropout(rate = 0.1))
# add second hidden layer
model.add(Dense(activation = 'relu', units = 6, kernel_initializer = 'uniform'))
model.add(Dropout(rate = 0.1))
#add output layer
model.add(Dense(activation = 'sigmoid', units = 1, kernel_initializer = 'uniform'))

#compile the ann
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# fit ann to training set
model.fit(X_train, y_train, batch_size = 40, epochs = 80)

#part 3
# Predicting the Test set results
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

newpred = model.predict(sc.transform(np.array([[600.0,0,1,40,3,60000,2,1,1,50000]])))
newpred = (newpred > 0.5)
print(newpred)
