# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

class ANN:
    def __init__(self):
        self.dataset = pd.read_csv('Churn_Modelling.csv')
        self.X = self.dataset.iloc[:, 3:13].values
        self.y = self.dataset.iloc[:, 13].values
        
        # Encoding categorical data
        labelencoder_X_1 = LabelEncoder()
        self.X[:, 1] = labelencoder_X_1.fit_transform(self.X[:, 1])
        labelencoder_X_2 = LabelEncoder()
        self.X[:, 2] = labelencoder_X_2.fit_transform(self.X[:, 2])
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = 0.2, random_state = 0)
        
        # Value Scaling
        self.sc = StandardScaler()
        self.X_train = self.sc.fit_transform(self.X_train)
        self.X_test = self.sc.transform(self.X_test)
    
    def build_model(self, optimiser = 'adam'):
        model = Sequential()
        model.add(Dense(activation = 'relu', units = 6, kernel_initializer = 'uniform', input_dim = 10))
        model.add(Dropout(rate = 0.1))
        model.add(Dense(activation = 'relu', units = 6, kernel_initializer = 'uniform'))
        model.add(Dropout(rate = 0.1))
        model.add(Dense(activation = 'sigmoid', units = 1, kernel_initializer = 'uniform'))
        model.compile(optimizer = optimiser, loss = 'binary_crossentropy', metrics = ['accuracy'])
        return model

    def run_kfold(self, batchSize = 40, eps = 80, k = 10):
        self.model = KerasClassifier(build_fn = self.build_model, batch_size = batchSize, epochs = eps)
        accuracies = cross_val_score(estimator = self.model, X = self.X_train, y = self.y_train, cv = k, n_jobs = -1)
        return accuracies
    
    def run_gridsearch(self, k = 10):
        self.model = KerasClassifier(build_fn = self.build_model)
        parameters = {'batch_size' : [20, 32, 60],
                      'epochs' : [80, 100, 500],
                      'optimiser' : ['adam','rmsprop']}
        gs = GridSearchCV(estimator = self.model, param_grid = parameters, scoring = 'accuracy', cv = k, n_jobs = -1)
        gs = gs.fit(X = self.X_train, y = self.y_train)
        best_params = gs.best_params_
        best_accuracy = gs.best_score_
        return (best_params, best_accuracy)
        
        
        
        
        