#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 22:51:51 2024

@author: alim
"""

import os
import cv2
from keras.models import load_model
from PIL import Image
import numpy as np
from natsort import natsorted
import matplotlib.pyplot as plt
import math
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


# Modeli Yükleme
model = load_model('BrainTumor150EpochsCategorical.keras')


# Klasördeki Tüm Dosyaları Tarama
directory = '/mnt/c/Users/alim/OneDrive/Masaüstü/archive/pred2'
tumor_count = 0
no_tumor_count = 0

Tumorlu = []
Tumorsuz = []

y_pred = []  # Tahminleri saklamak için liste
y_true = []  # Gerçek etiketleri saklamak için liste


for filename in natsorted(os.listdir(directory)):
    if filename.endswith('.jpg') or filename.endswith('.png'):  # Görüntü Dosyalarını Seç
        image_path = os.path.join(directory, filename)
    

        # Görüntüyü Yükle
        image = cv2.imread(image_path)
    
    
        # Görüntüyü Yeniden Boyutlandırma
        img = Image.fromarray(image)  
        img = img.resize((64,64))
        
        
        # Görüntüyü numpy dizisine çevirme
        img = np.array(img)
               
        
        # Bir boyut daha ekleme (3D --> 4D)
        input_img = np.expand_dims(img, axis=0)


        #Tahmin yapma
        result = model.predict(input_img)  # Predict metodunu kullan
        predicted_class = np.argmax(result, axis=1)   # En yüksek tahmin edilen sınıfı al
        
        # Tahminleri ve gerçek etiketleri saklama
        y_pred.append(predicted_class[0])  # Tahmin
        y_true.append(1 if 'y' in filename else 0)  # Gerçek etiket
        
        
        # Tümörlü veya Tümörsüz Olarak Sayımı Artırmanatsorted
        if predicted_class == 1:
            tumor_count += 1
            Tumorlu.append(filename)
        else:
            no_tumor_count += 1
            Tumorsuz.append(filename)

# Sonuçları Sıralama
Tumorlu = sorted(Tumorlu)  # Tümörlü resimleri dosya adına göre sıralama
Tumorsuz = sorted(Tumorsuz)  # Tümörsüz resimleri dosya adına göre sıralama

# Sonuçları Yazma
print(f"Tümörlü Resim Sayısı: {tumor_count}")
print(Tumorlu)

print(f"Tümörsüz Resim Sayısı: {no_tumor_count}")
print(Tumorsuz)

print("Dogru Degerler: \n", y_true)
print("Tahmin Edilen Degerler: \n", y_pred)

# Confusion Matrix Hesaplama
cm = confusion_matrix(y_true, y_pred)

# Görselleştirme
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Tumor', 'Tumor'], yticklabels=['No Tumor', 'Tumor'])
plt.ylabel('Gerçek')
plt.xlabel('Tahmin')
plt.title('Confusion Matrix')
plt.show()

# Sınıflandırma raporu
report = classification_report(y_true, y_pred, target_names=['No Tumor', 'Tumor'])
print(report)

# Özel hesaplamalar (Tumorlu --> P  || Tumorsuz --> N)
accuracy = (cm[0, 0] + cm[1, 1]) / np.sum(cm)
sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0])  # TP / (TP + FN)
specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])  # TN / (TN + FP)
f1_score = 2 * (sensitivity * (cm[1, 1] / (cm[1, 1] + cm[0, 1]))) / (sensitivity + (cm[1, 1] / (cm[1, 1] + cm[0, 1])))
#F1 = 2*(precision*sensitivity/precision+sensitivity)


print(f'Accuracy: {accuracy:.2f}')
print(f'Sensitivity: {sensitivity:.2f}')
print(f'Specificity: {specificity:.2f}')
print(f'F1 Score: {f1_score:.2f}')









# Her görüntüyü tahmin ile birlikte görselleştirme
def plot_images_with_predictions(directory, Tumorlu, Tumorsuz):
    images_to_plot = Tumorlu + Tumorsuz  # Her iki listeyi birleştir
    labels = ['Tumor' if img in Tumorlu else 'No Tumor' for img in images_to_plot]  # Etiketleri belirle
    
    # Toplam görüntü sayısına göre ızgara boyutunu ayarla
    num_images = len(images_to_plot)
    grid_size = math.ceil(math.sqrt(num_images))
    
    plt.figure(figsize=(15, 15))
    for i, filename in enumerate(images_to_plot):
        image_path = os.path.join(directory, filename)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        plt.subplot(grid_size, grid_size, i + 1)  # Dinamik ızgara boyutu kullan
        plt.imshow(image)
        plt.title(labels[i])
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()




# Tümörlü ve tümörsüz sayısını çubuk grafikle görselleştirme
def plot_tumor_counts(tumor_count, no_tumor_count):
    plt.figure(figsize=(7, 5))
    plt.bar(['Tumor', 'No Tumor'], [tumor_count, no_tumor_count], color=['red', 'green'])
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("Tumor vs No Tumor Count")
    plt.show()




# Çizimleri çalıştırma
plot_images_with_predictions(directory, Tumorlu, Tumorsuz)  # Tahminli görselleri çiz
plot_tumor_counts(tumor_count, no_tumor_count)  # Sayıları gösteren çubuk grafik




