# -*- coding: utf-8 -*-

from random import randint
import numpy as np
    
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

class ANN_ARITH:
    def generateData(self, size = 100):
        opf = [lambda m,n: m + n, lambda m,n: m * n]
        data = []
        for i in range(size):
            x = randint(0,15)
            y = randint(0,15)
            opi = randint(0,1)
            res = opf[opi](x, y)
            data.append(self.genBinary(x,4) + [opi] + self.genBinary(y,4) + self.genBinary(res))
        return data
    def genBinary(self, x, digits = 8):
        binary = []
        while x > 0:
            mod = int(x) % int(2)
            binary.append(mod)
            x -= mod
            x /= 2
            digits -= 1
        for i in range(digits):
            binary.append(0)
        return binary[::-1]
    def __init__(self, file = None):
        self.model = Sequential()
    
        self.model.add(Dense(activation = 'relu', units = 32, kernel_initializer = 'uniform', input_dim = 9))
        self.model.add(Dropout(rate=0.1))
        
        self.model.add(Dense(activation = 'relu', units = 32, kernel_initializer = 'uniform'))
        self.model.add(Dropout(rate=0.1))
        
        self.model.add(Dense(activation = 'relu', units = 32, kernel_initializer = 'uniform'))
        self.model.add(Dropout(rate=0.1))
        
        self.model.add(Dense(activation = 'relu', units = 32, kernel_initializer = 'uniform'))
        self.model.add(Dropout(rate=0.1))
        
        self.model.add(Dense(activation = 'sigmoid', units = 8, kernel_initializer = 'uniform'))
        
        self.model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        
        if file is not None:
            self.model.load_weights(file)
    
    def fit_gen(self, rows = 500000, batchSize = 8, epochs = 25):
        data = np.array(self.generateData(rows), dtype=np.int32)
        
        X = data[:,0:9]
        y = data[:,9:]
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
        self.model.fit(X_train, y_train, batch_size = batchSize, epochs = epochs)

        scores = self.model.evaluate(X_test, y_test)
        print("accuracy: ", scores[1] * 100)
    
    def evaluate(self, testSize = 100000):
        data = np.array(self.generateData(testSize), dtype=np.int32)
        
        X = data[:,0:9]
        y = data[:,9:]
        scores = self.model.evaluate(X, y)
        print("accuracy: ", scores[1] * 100)
    
    def prediction(self, x,y,o,r):
        test = np.array(self.genBinary(x,4) + [o] + self.genBinary(y,4)).reshape((1,9))
        yt = np.array(self.genBinary(r))
        
        res = self.model.predict(test)
        res = res > 0.65
        
        success = True
        for i,col in enumerate(res[0]):
            if (bool(col) and bool(yt[i])) or (not bool(col) and not bool(yt[i])):
                pass
            else:
                success = False
                break
        print(success)
        if not success:
            print(yt)
            print(res)
