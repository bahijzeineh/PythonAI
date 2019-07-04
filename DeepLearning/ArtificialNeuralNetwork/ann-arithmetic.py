# -*- coding: utf-8 -*-

from random import randint
import numpy as np

opf = [lambda m,n: m + n, lambda m,n: m * n]
def generateData(size = 100):
    data = []
    for i in range(size):
        x = randint(0,7)
        y = randint(0,7)
        opi = randint(0,1)
        res = opf[opi](x, y)
        data.append(genBinary(x) + [opi] + genBinary(y) + genBinary(res,64))
    return data
def genBinary(x, digits = 8):
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
data = np.array(generateData(10000), dtype=np.int32)

X = data[:,0:17]
y = data[:,17:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Value Scaling

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

model = Sequential()

model.add(Dense(activation = 'relu', units = 32, kernel_initializer = 'uniform', input_dim = 17))
model.add(Dropout(rate=0.1))

model.add(Dense(activation = 'relu', units = 32, kernel_initializer = 'uniform'))
model.add(Dropout(rate=0.2))

model.add(Dense(activation = 'sigmoid', units = 64, kernel_initializer = 'uniform'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# fit ann to training set
model.fit(X_train, y_train, batch_size = 40, epochs = 80)

scores = model.evaluate(X_test, y_test)
print("accuracy: ", scores[1] * 100)

def prediction(x,y,o,r):
    test = np.array(genBinary(x) + [o] + genBinary(y)).reshape((1,17))
    yt = genBinary(y, 64)
    
    res = model.predict(test)
    print(res == yt)
