#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 11:37:03 2019

@author: cemsezeroglu
"""

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers import Dense
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten

#Object

classifier = Sequential()

#First step - Convolution
classifier.add(Conv2D(32,3,3, input_shape=(64,64,3),activation='relu'))

#Secon Step - Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

#2.Convolution
classifier.add(Conv2D(32,3,3,activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Third Step - Flattening
classifier.add(Flatten())

#4th - Neural Networks
classifier.add(Dense(output_dim = 128,activation = 'relu'))
classifier.add(Dense(output_dim = 1,activation = 'sigmoid'))

#CNN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#CNN and IMAGES

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip= True)

test_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip= True)

training_set = train_datagen.flow_from_directory('/content/CNN-CAPSTONE/training_set.zip',
                                                 target_size=(64,64),
                                                 batch_size= 1,
                                                 class_mode='binary')
test_set = test_datagen.flow_from_directory('/content/CNN-CAPSTONE/test_set.zip',
                                            target_size=(64,64),
                                            batch_size=1,
                                            class_mode='binary')
#/Users/cemsezeroglu/Desktop/CNN/CNN_KANSER_VERİLERİ/training_set
#/Users/cemsezeroglu/Desktop/CNN/CNN_KANSER_VERİLERİ/test_set
classifier.fit_generator(training_set,
                         samples_per_epoch = 800,
                         nb_epoch = 3,
                         validation_data= test_set,
                         nb_val_samples = 2000)
import numpy as np
import pandas as pd

test_set.reset()
pred=classifier.predict_generator(test_set,verbose=1,steps = 80)


#pred = list(map(round,pred))
pred[pred > .5]=1
pred[pred <= .5] = 0

print("---------------------")

print('Prediction tamam')
#labels = (training_set.class_indicates)

test_labels = []

for i in range(0,int(80)):
    test_labels.extend(np.array(test_set[i][1]))
    
print('test_labels : ')
print(test_labels)


dosyaisimleri = test_set.filenames

sonuc= pd.DataFrame()
sonuc['dosyaisimleri'] = dosyaisimleri
sonuc['tahminler']=pred
sonuc['test']=test_labels

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_labels,pred)
print(cm)


    
