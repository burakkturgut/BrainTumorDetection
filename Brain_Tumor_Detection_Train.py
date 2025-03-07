#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 21:01:56 2024

@author: alim
"""

import cv2
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


image_directory = '/mnt/c/Users/burak/OneDrive/Masaüstü/archive/datasets/'

no_tumor_images = os.listdir(image_directory + 'no/')
yes_tumor_images = os.listdir(image_directory + 'yes/')

dataset = []  #Görüntülerin piksellerini depolamak için
label = []    #Görüntülere ait etiketleri depolamak için

INPUT_SIZE = 64
#print(no_tumor_images)


#path = 'no0.jpg'
#print(path.split('.')[1])


for i , image_name in enumerate(no_tumor_images):
    if(image_name.split('.')[1] == 'jpg'):
        image = cv2.imread(image_directory + 'no/' + image_name) # Görüntüleri okuduk.
        image = Image.fromarray(image, 'RGB')  # Resimleri RGB ye çevirdik.
        image = image.resize((INPUT_SIZE,INPUT_SIZE)) # Resimlerin boyutunu ayarladık.
        dataset.append(np.array(image))
        label.append(0) # Tumor yok demek
        
        
        
for i , image_name in enumerate(yes_tumor_images):
    if(image_name.split('.')[1] == 'jpg'):
        image = cv2.imread(image_directory + 'yes/' + image_name) # Görüntüleri okuduk.
        image = Image.fromarray(image, 'RGB')  # Resimleri RGB ye çevirdik.
        image = image.resize((INPUT_SIZE,INPUT_SIZE)) # Resimlerin boyutunu ayarladık.
        dataset.append(np.array(image))
        label.append(1) # Tumor var demek   
        
    
        
#print(dataset)
#print(len(label)) # 3000 tane veri var.


# Dataset ve Label listelerini NumPy array'lerine dönüştürme.
dataset = np.array(dataset)
label = np.array(label)



# Verileri Eğitim ve Test Olarak Bölme
x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=0)

x_train = normalize(x_train, axis=1)
x_test = normalize(x_test, axis=1)

# Etiketleri one-hot encoding formatına dönüştürme.
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)
#(Modelin softmax fonk. ile uyumlu olması için)




# Modelin Oluşturulması
model = Sequential()

model.add(Conv2D(32, (3,3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3))) #64,64,3
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D(32, (3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D(32, (3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))  #Overfittingi engellemek için
model.add(Dense(2))
model.add(Activation('softmax'))


"""
NOT:
    Binary CrossEntropy = 1, sigmoid         --> olmalı
    Categorical Cross Entryopy = 2, softmax  --> olmalı.
"""

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# Early Stopping callback'i oluştur
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)


history = model.fit(x_train, y_train,
    batch_size=16,
    epochs=150,  # Yüksek bir değer veriyoruz, erken durdurma ile daha önce duracaktır.
    validation_data=(x_test, y_test),
    shuffle=False,
    verbose=1,
    callbacks=[early_stopping]  # Early stopping callback'i ekliyoruz
    )


# Eğitim ve doğrulama kayıplarını çizdirme
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Eğitim ve doğrulama doğruluğunu çizdirme
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()



model.save('BrainTumor150EpochsCategorical.keras')