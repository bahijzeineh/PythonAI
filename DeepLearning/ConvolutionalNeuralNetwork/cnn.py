# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 07:28:01 2019

@author: Bahij
"""

# build CNN

import numpy as np
from keras.preprocessing import image

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten

from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator

imgd = 128

class CNN:
    def __init__(self, weightsFile = None, dropout = 0.2):
        self.model = Sequential()
        
        self.model.add(Convolution2D(
                strides = 1, filters = 32, 
                kernel_size = 3,  padding = 'same',
                input_shape = (imgd, imgd, 3), 
                activation = 'relu'))
        self.model.add(MaxPooling2D(pool_size = 2))
        
        self.model.add(Convolution2D(
                strides = 1, filters = 32, 
                kernel_size = 3, padding = 'same',
                activation = 'relu'))
        self.model.add(MaxPooling2D(pool_size = 2))
        
        self.model.add(Convolution2D(
                strides = 1, filters = 64, 
                kernel_size = 3,  padding = 'same',
                activation = 'relu'))
        self.model.add(MaxPooling2D(pool_size = 2))
        
        self.model.add(Convolution2D(
                strides = 1, filters = 64, 
                kernel_size = 3,  padding = 'same',
                activation = 'relu'))
        self.model.add(MaxPooling2D(pool_size = 2))

        self.model.add(Flatten())
        
        self.model.add(Dense(units = 64, activation = 'relu'))
        self.model.add(Dropout(rate = dropout))
        self.model.add(Dense(units = 64, activation = 'relu'))
        self.model.add(Dense(units = 64, activation = 'relu'))
        self.model.add(Dropout(rate = dropout/2))
        self.model.add(Dense(units = 1, activation = 'sigmoid'))
        
        optimizer = Adam(lr=1e-3)
        self.model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
        
        if weightsFile is not None:
            self.model.load_weights(weightsFile)

    def fit_gen(self):
        train_datagen = ImageDataGenerator(
                rescale = 1./255,
                shear_range = 0.2,
                zoom_range = 0.2,
                horizontal_flip = True)
        
        test_datagen = ImageDataGenerator(rescale = 1./255)
        
        train_generator = train_datagen.flow_from_directory(
                'dataset/training_set',
                target_size = (imgd, imgd),
                batch_size = 32,
                class_mode = 'binary')
        
        test_generator = test_datagen.flow_from_directory(
                'dataset/test_set',
                target_size = (imgd, imgd),
                batch_size = 32,
                class_mode = 'binary')
        
        self.model.fit_generator(
                train_generator,
                steps_per_epoch = 8000,
                epochs = 3,
                validation_data = test_generator,
                validation_steps = 2000)
        return test_generator.class_indices
        
    def pred(self, imgloc):
        img1 = image.load_img(imgloc, target_size = (imgd, imgd))
        img1 = image.img_to_array(img1)
        imgs = np.array([img1])
        
        return self.model.predict(imgs)
