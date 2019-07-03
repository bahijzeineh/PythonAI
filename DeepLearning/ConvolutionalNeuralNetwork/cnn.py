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
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten

from keras.preprocessing.image import ImageDataGenerator

class CNN:
    def __init__(self):
        self.model = Sequential()
        
        self.model.add(Convolution2D(
                strides = 1, filters = 32, 
                kernel_size = 3, 
                input_shape = (64, 64, 3), 
                activation = 'relu'))
        
        self.model.add(MaxPooling2D(pool_size = 2))
        
        self.model.add(Convolution2D(
                strides = 1, filters = 48, 
                kernel_size = 3, 
                activation = 'relu'))
        
        self.model.add(MaxPooling2D(pool_size = 2))
        
        self.model.add(Flatten())
        
        self.model.add(Dense(units = 128, activation = 'relu'))
        self.model.add(Dense(units = 1, activation = 'sigmoid'))
        
        self.model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    def fit_gen(self):
        train_datagen = ImageDataGenerator(
                rescale=1./255,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True)
        
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_directory(
                'dataset/training_set',
                target_size=(64, 64),
                batch_size=32,
                class_mode='binary')
        
        test_generator = test_datagen.flow_from_directory(
                'dataset/test_set',
                target_size=(64, 64),
                batch_size=32,
                class_mode='binary')
        
        self.model.fit_generator(
                train_generator,
                steps_per_epoch=8000,
                epochs=3,
                validation_data=test_generator,
                validation_steps=2000)
        return test_generator.class_indices
        
    def pred(self):
        img1 = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64,64))
        img1 = image.img_to_array(img1)
        img2 = image.load_img('dataset/single_prediction/cat_or_dog_2.jpg', target_size=(64,64))
        img2 = image.img_to_array(img2)
        imgs=np.array([img1,img2])
        
        return self.model.predict(imgs)

            