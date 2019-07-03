# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 07:28:01 2019

@author: Bahij
"""

# build CNN

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten

model = Sequential()

model.add(Convolution2D(
        strides = 1, filters = 32, 
        kernel_size = 3, 
        input_shape = (64, 64, 3), 
        activation = 'relu'))

model.add(MaxPooling2D(pool_size = 2))

model.add(Flatten())

model.add(Dense(units = 128, activation = 'relu'))
model.add(Dense(units = 1, activation = 'sigmoid'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

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

model.fit_generator(
        train_generator,
        steps_per_epoch=8000,
        epochs=5,
        validation_data=test_generator,
        validation_steps=2000)

