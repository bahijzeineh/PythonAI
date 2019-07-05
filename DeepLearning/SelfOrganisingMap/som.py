# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 14:51:13 2019

@author: Bahij
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler

dataset = pd.read_csv('Credit_Card_Applications.csv')

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

sc = MinMaxScaler()
X = sc.fit_transform(X)

som = MiniSom(x = 12, y = 12, input_len = 15)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o','s']
colors = ['red','green']

for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5, w[1] + 0.5, 
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markersize = 10,
         markeredgewidth = 2)


mappings = som.win_map(X)
frauds = mappings[(1,11)]
frauds = sc.inverse_transform(frauds)